from django.urls import reverse
from rest_framework import status
from api.user.models.user_models import User
from api.auth.models.auth_models import UserRefreshToken
from . import BaseMongoTestCase

class AuthenticationTests(BaseMongoTestCase):
    def setUp(self):
        super().setUp()
        self.registration_url = reverse('registration')
        self.login_url = reverse('login')
        self.logout_url = reverse('logout')
        self.refresh_url = reverse('refresh')
        
        self.valid_user_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'testpass123'
        }
        
        self.invalid_user_data = {
            'username': 'test',
            'email': 'invalid-email',
            'password': '123'
        }

    def tearDown(self):
        super().tearDown()
        User.objects.delete()
        UserRefreshToken.objects.delete()

    def test_user_registration_success(self):
        """Test successful user registration"""
        response = self.client.post(self.registration_url, self.valid_user_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(User.objects.count(), 1)
        self.assertEqual(User.objects.first().email, 'test@example.com')

    def test_user_registration_invalid_data(self):
        """Test registration with invalid data"""
        response = self.client.post(self.registration_url, self.invalid_user_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(User.objects.count(), 0)

    def test_user_registration_duplicate_email(self):
        """Test registration with duplicate email"""
        self.client.post(self.registration_url, self.valid_user_data, format='json')
        
        duplicate_data = self.valid_user_data.copy()
        duplicate_data['username'] = 'different_username'
        response = self.client.post(self.registration_url, duplicate_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(User.objects.count(), 1)

    def test_login_success(self):
        """Test successful login"""
        self.client.post(self.registration_url, self.valid_user_data, format='json')
        
        response = self.client.post(self.login_url, {
            'email': self.valid_user_data['email'],
            'password': self.valid_user_data['password']
        }, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('accessToken', response.cookies)
        self.assertIn('refreshToken', response.cookies)

    def test_login_invalid_credentials(self):
        """Test login with invalid credentials"""
        response = self.client.post(self.login_url, {
            'email': 'wrong@example.com',
            'password': 'wrongpass'
        }, format='json')
        
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_logout(self):
        """Test logout functionality"""
        self.client.post(self.registration_url, self.valid_user_data, format='json')
        login_response = self.client.post(self.login_url, {
            'email': self.valid_user_data['email'],
            'password': self.valid_user_data['password']
        }, format='json')
        
        refresh_token = login_response.cookies.get('refreshToken').value
        
        response = self.client.post(self.logout_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        
        self.assertEqual(UserRefreshToken.objects.filter(token=refresh_token).count(), 0)

    def test_token_refresh(self):
        """Test token refresh functionality"""
        self.client.post(self.registration_url, self.valid_user_data, format='json')
        login_response = self.client.post(self.login_url, {
            'email': self.valid_user_data['email'],
            'password': self.valid_user_data['password']
        }, format='json')
        
        refresh_token = login_response.cookies.get('refreshToken').value
        
        response = self.client.post(self.refresh_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('accessToken', response.cookies)
        self.assertIn('refreshToken', response.cookies)
        
        self.assertEqual(UserRefreshToken.objects.filter(token=refresh_token).count(), 0)

    def test_token_refresh_invalid_token(self):
        """Test token refresh with invalid token"""
        response = self.client.post(self.refresh_url)
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)