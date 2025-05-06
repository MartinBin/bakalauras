import logging
from rest_framework import status
from rest_framework.generics import RetrieveAPIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from api.user.serializers.user_serializers import UserSerializer

logger = logging.getLogger(__name__)

class UserView(RetrieveAPIView):
    serializer_class = UserSerializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        return self.request.user
    
    def get(self, request, *args, **kwargs):
        try:
            return self.retrieve(request, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error in user view: {str(e)}")
            return Response({'error': 'Authentication failed'}, status=status.HTTP_401_UNAUTHORIZED)