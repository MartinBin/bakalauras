import os
import sys
import torch
import logging
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, parser_classes, permission_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from django.conf import settings
from .models import PredictionResult, User
from .serializers import PredictionResultSerializer, UserRegistrationSerializer, UserLoginSerializer, UserSerializer
from rest_framework.routers import DefaultRouter
import cv2
from datetime import datetime
from torchvision import transforms
import uuid
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/")))

from Trainer import Trainer
from models.Unet import UNet
from models.Encoder import Encoder
from models.Decoder import Decoder

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
@permission_classes([AllowAny])
def predict(request):
    try:
        left_image = request.FILES.get('left_image')
        right_image = request.FILES.get('right_image')
        
        if not left_image or not right_image:
            return Response({'error': 'Both left and right images are required'}, status=400)
            
        logger.info(f"Received left image: {left_image.name}, right image: {right_image.name}")

        prediction = PredictionResult.objects.create(
            point_cloud_path='', 
            metadata={
                'left_image_name': left_image.name,
                'right_image_name': right_image.name
            },
            metrics={}
        )
        
        logger.info(f"Created prediction record with ID: {prediction.id}")

        left_temp_path, right_temp_path = save_temp_images(left_image, right_image)
        
        try:
            logger.info("Starting prediction process")
            result = run_prediction(left_temp_path, right_temp_path)
            
            logger.info(f"Prediction completed successfully: {result}")

            prediction.point_cloud_path = result['point_cloud_path']
            if 'visualization_path' in result:
                prediction.visualization_path = result['visualization_path']
            prediction.metrics = result.get('metrics', {})
            prediction.save()
            
            logger.info(f"Updated prediction record with results")
            
            return Response({
                'id': str(prediction.id),
                'point_cloud_path': prediction.point_cloud_path,
                'visualization_path': prediction.visualization_path,
                'metrics': prediction.metrics,
                'unet_outputs': result.get('unet_outputs', {})
            })
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            prediction.metadata['error'] = str(e)
            prediction.save()
            
            return Response({
                'error': f'Error during prediction: {str(e)}',
                'id': str(prediction.id)
            }, status=500)
            
        finally:
            try:
                os.remove(left_temp_path)
                os.remove(right_temp_path)
                logger.info("Temporary files removed successfully")
            except Exception as e:
                logger.warning(f"Failed to remove temporary files: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return Response({'error': f'Error processing images: {str(e)}'}, status=500)

class PredictionResultViewSet(viewsets.ModelViewSet):
    serializer_class = PredictionResultSerializer
    
    def get_queryset(self):
        return PredictionResult.objects.all()


def save_point_cloud_with_image_overlay(predicted_point_cloud, original_image, save_path, sample_index):
    """
    Save a point cloud with image overlay to a file.
    
    Args:
        predicted_point_cloud: The predicted point cloud (numpy array)
        original_image: The original image to overlay (numpy array)
        save_path: Base path to save the visualization
        sample_index: Index of the current sample
    """
    try:
        vis_dir = os.path.join(save_path, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        logger.info(f"Point cloud type: {type(predicted_point_cloud)}")
        logger.info(f"Original image type: {type(original_image)}")
        
        if torch.is_tensor(predicted_point_cloud):
            logger.info(f"Converting point cloud tensor to numpy. Shape: {predicted_point_cloud.shape}")
            predicted_point_cloud = predicted_point_cloud.detach().cpu().numpy()
        
        if torch.is_tensor(original_image):
            logger.info(f"Converting image tensor to numpy. Shape: {original_image.shape}")
            original_image = original_image.detach().cpu().numpy()
        
        logger.info(f"Point cloud shape after conversion: {predicted_point_cloud.shape}")
        logger.info(f"Original image shape after conversion: {original_image.shape}")
    
        if predicted_point_cloud.ndim > 2 and predicted_point_cloud.shape[0] > 1:
            logger.info(f"Handling batch dimension for point cloud. Original shape: {predicted_point_cloud.shape}")
            predicted_point_cloud = predicted_point_cloud[0]
            logger.info(f"Point cloud shape after batch handling: {predicted_point_cloud.shape}")
        
        logger.info(f"Image shape before processing: {original_image.shape}")
        
        if original_image.ndim == 4:
            logger.info("Image is 4D (batch, channels, height, width), extracting first batch")
            original_image = original_image[0]
            logger.info(f"Image shape after removing batch dimension: {original_image.shape}")
        
        if original_image.ndim == 3 and original_image.shape[0] <= 4:
            logger.info("Image appears to be in CHW format, transposing to HWC")
            try:
                original_image = np.transpose(original_image, (1, 2, 0))
                logger.info(f"Image shape after transposing: {original_image.shape}")
            except ValueError as e:
                logger.error(f"Error transposing image: {str(e)}")
                logger.error(f"Image shape: {original_image.shape}")
                logger.info("Creating a new image from scratch")
                h, w = original_image.shape[1], original_image.shape[2]
                new_image = np.zeros((h, w, 3), dtype=np.uint8)
                for c in range(min(3, original_image.shape[0])):
                    new_image[:, :, c] = original_image[c, :, :]
                original_image = new_image
                logger.info(f"New image shape: {original_image.shape}")
        
        if original_image.max() > 1.0:
            logger.info(f"Normalizing image. Original range: [{original_image.min()}, {original_image.max()}]")
            original_image = original_image / 255.0
            logger.info(f"Image range after normalization: [{original_image.min()}, {original_image.max()}]")
        
        if predicted_point_cloud.ndim > 2:
            logger.info(f"Reshaping point cloud. Original shape: {predicted_point_cloud.shape}")
            predicted_point_cloud = predicted_point_cloud.reshape(-1, 3)
            logger.info(f"Point cloud shape after reshaping: {predicted_point_cloud.shape}")
        
        logger.info("Creating Open3D point cloud")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(predicted_point_cloud)
        
        logger.info("Creating visualization window")
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        
        vis.add_geometry(pcd)
        
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        
        logger.info("Rendering point cloud")
        vis.poll_events()
        vis.update_renderer()
        
        logger.info("Capturing rendered image")
        rendered_image = vis.capture_screen_float_buffer(do_render=True)
        rendered_image = np.asarray(rendered_image)
        logger.info(f"Rendered image shape: {rendered_image.shape}")
        
        vis.destroy_window()
        
        logger.info("Creating matplotlib figure")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        logger.info("Plotting original image")
        logger.info(f"Final image shape for matplotlib: {original_image.shape}")
        ax1.imshow(original_image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        logger.info("Plotting rendered point cloud")
        ax2.imshow(rendered_image)
        ax2.set_title('Point Cloud Visualization')
        ax2.axis('off')
        
        logger.info("Saving figure")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'sample_{sample_index}_overlay.png'), dpi=300)
        plt.close()
        
        try:
            logger.info("Creating colored point cloud")
            z_values = predicted_point_cloud[:, 2]
            z_min, z_max = z_values.min(), z_values.max()
            
            z_normalized = (z_values - z_min) / (z_max - z_min + 1e-8)
            
            colors = plt.cm.viridis(z_normalized)[:, :3]
            
            colored_pcd = o3d.geometry.PointCloud()
            colored_pcd.points = o3d.utility.Vector3dVector(predicted_point_cloud)
            colored_pcd.colors = o3d.utility.Vector3dVector(colors)
            
            logger.info("Saving colored point cloud")
            o3d.io.write_point_cloud(
                os.path.join(vis_dir, f'sample_{sample_index}_colored.ply'),
                colored_pcd
            )
        except Exception as e:
            logger.error(f"Error saving colored point cloud: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        logger.info(f"Saved visualization for sample {sample_index} to {vis_dir}")
        return True
    except Exception as e:
        logger.error(f"Error in save_point_cloud_with_image_overlay: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def run_prediction(left_image, right_image):
    try:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        
        left_img = Image.open(left_image).convert('RGB')
        right_img = Image.open(right_image).convert('RGB')

        left_img=transform(left_img).unsqueeze(0)
        right_img=transform(right_img).unsqueeze(0)
        
        model_location = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/Trained_Models"))
        logger.info(f"Loading models from {model_location}")
        
        trainer = Trainer(model_location=model_location,verbose=1)
        
        trainer.load_model()
        
        point_cloud_filename = f'point_cloud_{uuid.uuid4()}.ply'
        point_cloud_path = os.path.join(settings.MEDIA_ROOT, 'point_clouds', point_cloud_filename)
        os.makedirs(os.path.dirname(point_cloud_path), exist_ok=True)
        
        logger.info("Prediction is starting")
        point_cloud, metrics = trainer.predict(left_img, right_img, save_path=point_cloud_path)
        
        if point_cloud is None:
            raise ValueError("Failed to generate point cloud")
        
        logger.info(f"Point cloud shape: {point_cloud.shape}")
        logger.info(f"Point cloud type: {type(point_cloud)}")
        
        overlay_dir = os.path.join(settings.MEDIA_ROOT, 'point_clouds')
        os.makedirs(overlay_dir, exist_ok=True)
        
        visualization_success = False
        try:
            visualization_success = save_point_cloud_with_image_overlay(
                        predicted_point_cloud=point_cloud,
                        original_image=left_img,
                        save_path=overlay_dir,
                        sample_index=0
                    )
            if not visualization_success:
                logger.warning("Failed to create point cloud visualization")
        except Exception as e:
            logger.error(f"Error creating point cloud visualization: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        actual_point_cloud_filename = f"{point_cloud_filename}_predicted.ply"
        actual_point_cloud_path = os.path.join(settings.MEDIA_ROOT, 'point_clouds', actual_point_cloud_filename)
        logger.info(f"Actual point cloud path: {actual_point_cloud_path}")
        
        relative_point_cloud_path = os.path.join('media', 'point_clouds', actual_point_cloud_filename)
        logger.info(f"Relative point cloud path: {relative_point_cloud_path}")
        
        visualization_path = None
        if visualization_success:
            visualization_path = os.path.join('media', 'point_clouds', 'visualizations', f'sample_0_overlay.png')
        
        logger.info("Starting getting unet outputs")
        left_unet,right_unet = trainer.getUnetOutput(left_img, right_img)
        
        unet_output_paths = {}
        if left_unet is not None and right_unet is not None:
            unet_dir = os.path.join(settings.MEDIA_ROOT, 'unet_outputs')
            os.makedirs(unet_dir, exist_ok=True)
            
            if torch.is_tensor(left_unet):
                left_unet = left_unet.detach().cpu().numpy()
            if torch.is_tensor(right_unet):
                right_unet = right_unet.detach().cpu().numpy()
                
            logger.info(f"Left UNet shape before processing: {left_unet.shape}")
            logger.info(f"Right UNet shape before processing: {right_unet.shape}")
            
            left_unet = np.transpose(left_unet[0], (1, 2, 0))
            right_unet = np.transpose(right_unet[0], (1, 2, 0))
            
            left_unet = np.clip(left_unet * 255, 0, 255).astype(np.uint8)
            right_unet = np.clip(right_unet * 255, 0, 255).astype(np.uint8)
            
            logger.info(f"Left UNet shape after processing: {left_unet.shape}")
            logger.info(f"Right UNet shape after processing: {right_unet.shape}")
            
            left_unet_img = Image.fromarray(left_unet)
            right_unet_img = Image.fromarray(right_unet)
            
            left_unet_filename = f'left_unet_{uuid.uuid4()}.png'
            right_unet_filename = f'right_unet_{uuid.uuid4()}.png'
            
            left_unet_path = os.path.join(unet_dir, left_unet_filename)
            right_unet_path = os.path.join(unet_dir, right_unet_filename)
            
            left_unet_img.save(left_unet_path)
            right_unet_img.save(right_unet_path)
            
            relative_left_unet_path = os.path.join('media', 'unet_outputs', left_unet_filename)
            relative_right_unet_path = os.path.join('media', 'unet_outputs', right_unet_filename)
            
            unet_output_paths = {
                'left': relative_left_unet_path,
                'right': relative_right_unet_path
            }
        
        result = {
            'point_cloud_path': relative_point_cloud_path,
            'unet_outputs': unet_output_paths,
            'metrics': metrics or {}
        }
        
        if visualization_path:
            result['visualization_path'] = visualization_path
            
        return result
        
    except Exception as e:
        logger.error(f"Error in run_prediction: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def save_temp_images(left_image, right_image):
    """
    Save uploaded images temporarily and return their paths.
    """
    try:
        temp_dir = os.path.join(settings.MEDIA_ROOT, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        
        left_temp_path = os.path.join(temp_dir, f'left_input_{uuid.uuid4()}.png')
        right_temp_path = os.path.join(temp_dir, f'right_input_{uuid.uuid4()}.png')
        
        with open(left_temp_path, 'wb') as f:
            f.write(left_image.read())
        with open(right_temp_path, 'wb') as f:
            f.write(right_image.read())
            
        left_image.seek(0)
        right_image.seek(0)
        
        return left_temp_path, right_temp_path
        
    except Exception as e:
        logger.error(f"Error saving temporary images: {str(e)}")
        raise

@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    logger.info(f"Registration attempt with data: {request.data}")
    serializer = UserRegistrationSerializer(data=request.data)
    if serializer.is_valid():
        try:
            logger.info("Registration data is valid, creating user")
            user = serializer.save()
            logger.info(f"User created successfully: {user.username}")
            refresh = RefreshToken.for_user(user)
            logger.info("Refresh token generated")
            return Response({
                'user': {
                    'id': str(user.id),
                    'email': user.email,
                    'username': user.username
                },
                'refresh': str(refresh),
                'access': str(refresh.access_token),
            }, status=status.HTTP_201_CREATED)
        except Exception as e:
            logger.error(f"Error during user creation: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return Response({'error': f'Error creating user: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    logger.error(f"Validation errors: {serializer.errors}")
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    serializer = UserLoginSerializer(data=request.data)
    if serializer.is_valid():
        email = serializer.validated_data['email']
        password = serializer.validated_data['password']
        
        try:
            user = User.objects.get(email=email)
            if user.check_password(password):
                refresh = RefreshToken.for_user(user)
                return Response({
                    'refreshToken': str(refresh),
                    'accessToken': str(refresh.access_token),
                    'user': {
                        'id': str(user.id),
                        'email': user.email,
                        'username': user.username
                    }
                })
        except User.DoesNotExist:
            pass
            
        return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)
    logger.error(f"Validation errors: {serializer.errors}")
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user(request):
    try:
        serializer = UserSerializer(request.user)
        return Response(serializer.data)
    except Exception as e:
        logger.error(f"Error in user view: {str(e)}")
        return Response({'error': 'Authentication failed'}, status=status.HTTP_401_UNAUTHORIZED)