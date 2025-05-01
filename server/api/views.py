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
@permission_classes([IsAuthenticated])
def predict(request):
    try:
        left_image = request.FILES.get('left_image')
        right_image = request.FILES.get('right_image')
        
        if not left_image or not right_image:
            return Response({'error': 'Both left and right images are required'}, status=400)

        if not request.user.is_authenticated:
            return Response({'error': 'Authentication required'}, status=401)
            
        logger.info(f"Received left image: {left_image.name}, right image: {right_image.name}")


        left_image_path, right_image_path, relative_left_image_path, relative_right_image_path = save_user_images(request.user, left_image, right_image)

        prediction = PredictionResult.objects.create(
            user=request.user,
            point_cloud_path='', 
            metadata={
                'left_image_path': relative_left_image_path,
                'right_image_path': relative_right_image_path
            },
            metrics={}
        )
    
        logger.info(f"Created prediction record with ID: {prediction.id}")
        
        try:
            logger.info("Starting prediction process")
            result = run_prediction(left_image_path, right_image_path)
            

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
                'unet_outputs': result.get('unet_outputs', {}),
                'depth_values': result.get('depth_values',{}),
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
        
    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return Response({'error': f'Error processing images: {str(e)}'}, status=500)

class PredictionResultViewSet(viewsets.ModelViewSet):
    serializer_class = PredictionResultSerializer
    
    def get_queryset(self):
        return PredictionResult.objects.all()

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
        
        trainer = Trainer(model_location=model_location,verbose=0)
        
        trainer.load_model()
        
        point_cloud_filename = f'point_cloud_{uuid.uuid4()}.ply'
        point_cloud_path = os.path.join(settings.MEDIA_ROOT, 'point_clouds', point_cloud_filename)
        os.makedirs(os.path.dirname(point_cloud_path), exist_ok=True)
        
        logger.info("Prediction is starting")
        point_cloud, _ = trainer.predict(left_img, right_img, save_path=point_cloud_path)
        
        if point_cloud is None:
            raise ValueError("Failed to generate point cloud")
        
        logger.info(f"Point cloud shape: {point_cloud.shape}")
        logger.info(f"Point cloud type: {type(point_cloud)}")
        
        if torch.is_tensor(point_cloud):
            point_cloud_np = point_cloud.detach().cpu().numpy()
        else:
            point_cloud_np = point_cloud
            
        if point_cloud_np.ndim > 2:
            point_cloud_np = point_cloud_np.reshape(-1, 3)
            
        center = np.mean(point_cloud_np, axis=0)
        points_centered = point_cloud_np - center
        
        mse = np.mean(np.sum(points_centered ** 2, axis=1))
        
        mae = np.mean(np.abs(points_centered))
        
        num_points = len(points_centered)
        sample_size = min(1000, num_points)
        indices = np.random.choice(num_points, sample_size, replace=False)
        sampled_points = points_centered[indices]
        
        batch_size = 100
        max_dist = 0
        for i in range(0, sample_size, batch_size):
            batch = sampled_points[i:i+batch_size]
            distances = np.sqrt(np.sum((batch[:, np.newaxis] - points_centered) ** 2, axis=2))
            batch_max = np.max(distances)
            max_dist = max(max_dist, batch_max)
        
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'chamfer': float(max_dist)
        }
        
        logger.info(f"Calculated metrics: {metrics}")
        
        overlay_dir = os.path.join(settings.MEDIA_ROOT, 'point_clouds')
        os.makedirs(overlay_dir, exist_ok=True)
        
        visualization_success = False
        
        actual_point_cloud_filename = f"{point_cloud_filename}_predicted.ply"
        actual_point_cloud_path = os.path.join(settings.MEDIA_ROOT, 'point_clouds', actual_point_cloud_filename)
        logger.info(f"Actual point cloud path: {actual_point_cloud_path}")
        
        relative_point_cloud_path = os.path.join('media', 'point_clouds', actual_point_cloud_filename)
        logger.info(f"Relative point cloud path: {relative_point_cloud_path}")
        
        visualization_path = None
        if visualization_success:
            visualization_path = os.path.join('media', 'point_clouds', 'visualizations', f'sample_0_overlay.png')
        
        logger.info("Starting getting unet outputs")
        left_unet, right_unet, left_depth, right_depth = trainer.getUnetOutput(left_img, right_img)
        
        unet_output_paths = {}
        depth_values = {}
        
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
            
            if left_depth is not None and right_depth is not None:
                if torch.is_tensor(left_depth):
                    left_depth = left_depth.detach().cpu().numpy()
                if torch.is_tensor(right_depth):
                    right_depth = right_depth.detach().cpu().numpy()
                
                depth_values = {
                    'left': left_depth.flatten().tolist(),
                    'right': right_depth.flatten().tolist()
                }
        
        result = {
            'point_cloud_path': relative_point_cloud_path,
            'unet_outputs': unet_output_paths,
            'depth_values': depth_values,
            'metrics': metrics
        }
        
        if visualization_path:
            result['visualization_path'] = visualization_path
            
        return result
        
    except Exception as e:
        logger.error(f"Error in run_prediction: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def save_user_images(user, left_image, right_image):
    """
    Save uploaded images with unique user ID and return their paths.
    """
    try:
        user_images_dir = os.path.join(settings.MEDIA_ROOT, 'user_images',str(user.id))
        os.makedirs(user_images_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        left_image_name = f'{user.id}_left_{timestamp}_{uuid.uuid4()}.png'
        right_image_name = f'{user.id}_right_{timestamp}_{uuid.uuid4()}.png'
        
        left_image_path = os.path.join(user_images_dir, left_image_name)
        right_image_path = os.path.join(user_images_dir, right_image_name)

        relative_left_image_path = os.path.join('media','user_images',str(user.id),left_image_name)
        relative_right_image_path = os.path.join('media','user_images',str(user.id),right_image_name)

        with open(left_image_path, 'wb') as f:
            f.write(left_image.read())
        with open(right_image_path, 'wb') as f:
            f.write(right_image.read())
        
        left_image.seek(0)
        right_image.seek(0)
        
        return left_image_path, right_image_path, relative_left_image_path, relative_right_image_path
        
    except Exception as e:
        logger.error(f"Error saving user images: {str(e)}")
        raise ValidationError("Failed to save images.")

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

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_prediction_history(request):
    user = request.user
    predictions = PredictionResult.objects.filter(user=user).order_by('-timestamp')
    serializer = PredictionResultSerializer(predictions, many=True)
    return Response(serializer.data)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_prediction(request, prediction_id):
    try:
        prediction = PredictionResult.objects.get(id=prediction_id, user=request.user)

        file_paths = []
        if prediction.point_cloud_path:
            file_paths.append(os.path.join(settings.BASE_DIR, prediction.point_cloud_path))
        if getattr(prediction, 'visualization_path', None):
            file_paths.append(os.path.join(settings.BASE_DIR, prediction.visualization_path))
        
        for file_path in file_paths:
            if os.path.exists(file_path) and settings.MEDIA_ROOT in file_path:
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete file {file_path}: {str(e)}")
        
        prediction.delete()
        return Response({'success': True})
    except PredictionResult.DoesNotExist:
        return Response({'error': 'Prediction not found'}, status=404)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_all_user_predictions(request):
    try:
        predictions = PredictionResult.objects.filter(user=request.user)

        if not predictions:
            return Response({'message': 'No predictions found for this user.'}, status=404)

        file_paths = []
        for prediction in predictions:
            if prediction.point_cloud_path:
                file_paths.append(os.path.join(settings.BASE_DIR, prediction.point_cloud_path))
            if getattr(prediction, 'visualization_path', None):
                file_paths.append(os.path.join(settings.BASE_DIR, prediction.visualization_path))

        for file_path in file_paths:
            if os.path.exists(file_path) and settings.MEDIA_ROOT in file_path:
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete file {file_path}: {str(e)}")

        predictions.delete()

        return Response({'success': True, 'message': 'All predictions have been deleted successfully.'})
    except Exception as e:
        logger.error(f"Error deleting predictions: {str(e)}")
        return Response({'error': 'An error occurred while deleting predictions.'}, status=500)
