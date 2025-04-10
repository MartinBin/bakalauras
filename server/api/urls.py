from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import predict, register, login, user, PredictionResultViewSet

router = DefaultRouter()
router.register(r'prediction-results', PredictionResultViewSet, basename='prediction-result')

urlpatterns = [
    path('auth/registration/', register, name='register'),
    path('auth/login/', login, name='login'),
    path('auth/user/', user, name='user'),
    path('predict/', predict, name='predict'),
    path('', include(router.urls)),
]
