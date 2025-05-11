from django.urls import reverse
from rest_framework import status
from api.user.models.user_models import User
from . import BaseMongoTestCase

class UserTests(BaseMongoTestCase):
    def setUp(self):
        super().setUp()
        self.user_url = reverse('user')
        self.registration_url = reverse('registration')
        self.login_url = reverse('login')
        
        self.user_data = {
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'testpass123'
        }
        
        User.objects.delete()
        
        self.client.post(self.registration_url, self.user_data, format='json')
        login_response = self.client.post(self.login_url, {
            'email': self.user_data['email'],
            'password': self.user_data['password']
        }, format='json')
        
        self.client.cookies = login_response.cookies

    def tearDown(self):
        super().tearDown()
        User.objects.delete()

    def test_get_user_profile(self):
        """Test retrieving user profile"""
        response = self.client.get(self.user_url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['email'], self.user_data['email'])
        self.assertEqual(response.data['username'], self.user_data['username'])

    def test_get_user_profile_unauthorized(self):
        """Test retrieving user profile without authentication"""
        self.client.cookies.clear()
        
        response = self.client.get(self.user_url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)

    def test_update_user_profile(self):
        """Test updating user profile"""
        update_data = {
            'username': 'updated_username',
            'email': 'updated@example.com'
        }
        
        response = self.client.patch(self.user_url, update_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data['username'], update_data['username'])
        self.assertEqual(response.data['email'], update_data['email'])

    def test_update_user_profile_invalid_data(self):
        """Test updating user profile with invalid data"""
        invalid_data = {
            'username': '',
            'email': 'invalid-email'
        }
        
        response = self.client.patch(self.user_url, invalid_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_update_user_profile_duplicate_email(self):
        """Test updating user profile with duplicate email"""
        another_user = {
            'username': 'another_user',
            'email': 'another@example.com',
            'password': 'testpass123'
        }
        self.client.post(self.registration_url, another_user, format='json')
        
        update_data = {
            'username': self.user_data['username'],
            'email': another_user['email']
        }
        
        response = self.client.patch(self.user_url, update_data, format='json')
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_delete_user_account(self):
        """Test deleting user account"""
        response = self.client.delete(self.user_url)
        self.assertEqual(response.status_code, status.HTTP_204_NO_CONTENT)
        self.assertEqual(User.objects.count(), 0)

    def test_delete_user_account_unauthorized(self):
        """Test deleting user account without authentication"""
        self.client.cookies.clear()
        
        response = self.client.delete(self.user_url)
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertEqual(User.objects.count(), 1)
