import os
from rest_framework import status
from api.user.models.user_models import User
from api.prediction.models.prediction_models import PredictionResult
from django.core.files.uploadedfile import SimpleUploadedFile
from PIL import Image
import io
from . import BaseMongoTestCase

class PredictionTests(BaseMongoTestCase):
    def setUp(self):
        super().setUp()
        User.objects.delete()
        PredictionResult.objects.delete()
        
        self.user = User.objects.create(
            username=f'testuser_{id(self)}',
            email=f'test_{id(self)}@example.com',
            password='testpass123'
        )
        self.client.force_authenticate(user=self.user)
        
        self.left_image = self.create_test_image('left.jpg')
        self.right_image = self.create_test_image('right.jpg')

    def tearDown(self):
        super().tearDown()
        User.objects.delete()
        PredictionResult.objects.delete()

    def create_test_image(self, name):
        file = io.BytesIO()
        image = Image.new('RGB', (100, 100), 'white')
        image.save(file, 'png')
        file.name = name
        file.seek(0)
        return file

    def test_get_user_predictions(self):
        """Test retrieving user's prediction history"""
        response = self.client.get('/api/user/predictions/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIsInstance(response.data, list)

    def test_get_user_predictions_unauthorized(self):
        """Test retrieving predictions without authentication"""
        self.client.force_authenticate(user=None)
        response = self.client.get('/api/user/predictions/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_delete_all_user_predictions_unauthorized(self):
        """Test deleting all predictions without authentication"""
        self.client.force_authenticate(user=None)
        response = self.client.delete('/api/user/predictions/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_delete_all_user_predictions(self):
        """Test deleting all predictions for authenticated user"""
        PredictionResult.objects.create(
            user=self.user,
            point_cloud_path='test/path/point_cloud.ply',
            left_unet_path='test/path/left_unet.png',
            right_unet_path='test/path/right_unet.png',
            metadata={
                'left_image_path': 'test/path/left.png',
                'right_image_path': 'test/path/right.png'
            },
            metrics={'score': 0.85}
        )
        
        response = self.client.delete('/api/user/predictions/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(PredictionResult.objects.count(), 0)

    def test_delete_specific_prediction(self):
        """Test deleting a specific prediction"""
        prediction = PredictionResult.objects.create(
            user=self.user,
            point_cloud_path='test/path/point_cloud.ply',
            left_unet_path='test/path/left_unet.png',
            right_unet_path='test/path/right_unet.png',
            metadata={
                'left_image_path': 'test/path/left.png',
                'right_image_path': 'test/path/right.png'
            },
            metrics={'score': 0.85}
        )
        
        response = self.client.delete(f'/api/user/predictions/{prediction.id}/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(PredictionResult.objects.count(), 0)

    def test_delete_specific_prediction_unauthorized(self):
        """Test deleting a specific prediction without authentication"""
        prediction = PredictionResult.objects.create(
            user=self.user,
            point_cloud_path='test/path/point_cloud.ply',
            left_unet_path='test/path/left_unet.png',
            right_unet_path='test/path/right_unet.png',
            metadata={
                'left_image_path': 'test/path/left.png',
                'right_image_path': 'test/path/right.png'
            },
            metrics={'score': 0.85}
        )
        
        self.client.force_authenticate(user=None)
        response = self.client.delete(f'/api/user/predictions/{prediction.id}/')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
