from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import InvalidToken
from rest_framework_simplejwt.settings import api_settings
from .models import User

class MongoEngineJWTAuthentication(JWTAuthentication):
    """
    Custom JWT authentication backend for MongoEngine models.
    """
    def get_user(self, validated_token):
        """
        Attempt to get the user from the token.
        """
        try:
            user_id = validated_token[api_settings.USER_ID_CLAIM]
            user = User.objects.get(id=user_id)
            return user
        except User.DoesNotExist:
            raise InvalidToken('User not found')
        except Exception as e:
            raise InvalidToken(f'Error retrieving user: {str(e)}') 