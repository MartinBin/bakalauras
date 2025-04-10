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

# Set up logger
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/")))

from Trainer import Trainer
from models.Unet import UNet
from models.Encoder import Encoder
from models.Decoder import Decoder

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
@permission_classes([AllowAny])  # Allow any user to access this endpoint
def predict(request):
    try:
        left_image = request.FILES.get('left_image')
        right_image = request.FILES.get('right_image')
        
        if not left_image or not right_image:
            return Response({'error': 'Both left and right images are required'}, status=400)
            

        prediction = PredictionResult.objects.create(
            point_cloud_path='', 
            metadata={
                'left_image_name': left_image.name,
                'right_image_name': right_image.name
            },
            metrics={}
        )
        

        left_temp_path, right_temp_path = save_temp_images(left_image, right_image)
        
        try:

            result = run_prediction(left_temp_path, right_temp_path)
            

            prediction.point_cloud_path = result['point_cloud_path']
            prediction.visualization_path = result.get('visualization_path')
            prediction.metrics = result.get('metrics', {})
            prediction.save()
            
            return Response({
                'id': str(prediction.id),
                'point_cloud_path': prediction.point_cloud_path,
                'visualization_path': prediction.visualization_path,
                'metrics': prediction.metrics,
                'unet_outputs': result.get('unet_outputs', {})
            })
            
        finally:

            try:
                os.remove(left_temp_path)
                os.remove(right_temp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary files: {str(e)}")
        
    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        return Response({'error': 'Error processing images. Please try again.'}, status=500)

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
        
        trainer = Trainer(model_location=model_location)
        
        trainer.load_model()
        
        point_cloud_filename = f'point_cloud_{uuid.uuid4()}.ply'
        point_cloud_path = os.path.join(settings.MEDIA_ROOT, 'point_clouds', point_cloud_filename)
        os.makedirs(os.path.dirname(point_cloud_path), exist_ok=True)
        
        logger.info("Prediction is starting")
        point_cloud, metrics = trainer.predict(left_img, right_img, save_path=point_cloud_path)
        
        if point_cloud is None:
            raise ValueError("Failed to generate point cloud")
        
        actual_point_cloud_filename = f"{point_cloud_filename}_predicted.ply"
        actual_point_cloud_path = os.path.join(settings.MEDIA_ROOT, 'point_clouds', actual_point_cloud_filename)
        logger.info(f"Actual point cloud path: {actual_point_cloud_path}")
        
        relative_point_cloud_path = os.path.join('media', 'point_clouds', actual_point_cloud_filename)
        logger.info(f"Relative point cloud path: {relative_point_cloud_path}")
        
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
            
            # Convert to relative paths for the API response
            relative_left_unet_path = os.path.join('media', 'unet_outputs', left_unet_filename)
            relative_right_unet_path = os.path.join('media', 'unet_outputs', right_unet_filename)
            
            unet_output_paths = {
                'left': relative_left_unet_path,
                'right': relative_right_unet_path
            }
        
        return {
            'point_cloud_path': relative_point_cloud_path,
            'unet_outputs': unet_output_paths,
            'metrics': metrics or {}
        }
        
    except Exception as e:
        logger.error(f"Error in run_prediction: {str(e)}")
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
    serializer = UserRegistrationSerializer(data=request.data)
    if serializer.is_valid():
        try:
            user = serializer.save()
            refresh = RefreshToken.for_user(user)
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