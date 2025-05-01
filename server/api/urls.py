from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import predict, register, login, user, PredictionResultViewSet, user_prediction_history, delete_prediction, delete_all_user_predictions

router = DefaultRouter()
router.register(r'prediction-results', PredictionResultViewSet, basename='prediction-result')

urlpatterns = [
    path('auth/registration/', register, name='register'),
    path('auth/login/', login, name='login'),
    path('auth/user/', user, name='user'),
    path('predict/', predict, name='predict'),
    path('user/predictions/history', user_prediction_history, name='user_prediction_history'),
    path('user/predictions/', delete_all_user_predictions, name='delete_all_user_predictions'),
    path('user/predictions/<str:prediction_id>/', delete_prediction, name='delete_prediction'),
    path('', include(router.urls)),
]
