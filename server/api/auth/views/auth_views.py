from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken
from api.auth.serializers.auth_serializers import UserLoginSerializer, UserRegistrationSerializer
from api.user.models.user_models import User
from api.auth.models.auth_models import UserRefreshToken
import logging

logger = logging.getLogger(__name__)

class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = UserLoginSerializer(data=request.data)
        
        if serializer.is_valid():
            email = serializer.validated_data['email']
            password = serializer.validated_data['password']
            
            try:
                user = User.objects.get(email=email)
                if user.check_password(password):
                    refresh = RefreshToken.for_user(user)
                    access_token = str(refresh.access_token)
                    refresh_token = str(refresh)

                    try:
                        token_record = UserRefreshToken.objects.get(user=user)
                        token_record.delete()
                    except UserRefreshToken.DoesNotExist:
                        pass

                    UserRefreshToken.objects.create(user=user, token=str(refresh))

                    response = Response({'message': "Login successful"})
                
                    response.set_cookie(
                        key='accessToken',
                        value=access_token,
                        httponly=True,
                        secure=False,
                        samesite='Lax',
                        max_age=60 * 5
                    )
                    response.set_cookie(
                        key='refreshToken',
                        value=refresh_token,
                        httponly=True,
                        secure=False,
                        samesite='Lax',
                        max_age=60 * 60 * 24 * 7
                    )

                    return response
            except User.DoesNotExist:
                pass
                
            return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)

        logger.error(f"Validation errors: {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        refresh_token = request.COOKIES.get('refreshToken')
        if refresh_token:
            UserRefreshToken.objects.filter(token=refresh_token).delete()

        response = Response({'message': 'Logged out successfully'}, status=status.HTTP_200_OK)

        response.delete_cookie('accessToken')
        response.delete_cookie('refreshToken')

        return response
    
class RefreshTokenView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        refresh_token = request.COOKIES.get('refreshToken')
        if not refresh_token:
            return Response({'detail': 'No refresh token provided'}, status=400)

        try:
            token_record = UserRefreshToken.objects.get(token=refresh_token)
            token = RefreshToken(refresh_token)
            
            user_id = token['user_id']
            user = User.objects.get(id=user_id)

            access_token = token.access_token
            new_refresh_token = RefreshToken.for_user(user)

            token_record.delete()

            UserRefreshToken.objects.create(user=user, token=str(new_refresh_token))

            response = Response({'message': 'Token refreshed'})

            response.set_cookie(
                key='accessToken',
                value=str(access_token),
                httponly=True,
                secure=False,
                samesite='Lax',
                max_age=60 * 5
            )
            response.set_cookie(
                key='refreshToken',
                value=str(new_refresh_token),
                httponly=True,
                secure=False,
                samesite='Lax',
                max_age=60 * 60 * 24 * 7
            )

            return response

        except UserRefreshToken.DoesNotExist:
            return Response({'error': 'Invalid or expired refresh token'}, status=status.HTTP_401_UNAUTHORIZED)
        except Exception as e:
            logger.error(f"Error in refresh: {str(e)}")
            return Response({'error': 'Generating token failed'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
class RegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        logger.info(f"Registration attempt with data: {request.data}")
        serializer = UserRegistrationSerializer(data=request.data)
        
        if not serializer.is_valid():
            logger.error(f"Validation errors: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
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
