from django.urls import path, include
from rest_framework.routers import DefaultRouter
from api.prediction.views.prediction_view import PredictionView, UserPredictionDetailView, UserPredictionsView, PredictionResultViewSet
from api.user.views.user_views import UserView
from api.auth.views.auth_views import LoginView, LogoutView, RefreshTokenView, RegisterView

router = DefaultRouter()
router.register(r'prediction-results', PredictionResultViewSet, basename='prediction-result')

urlpatterns = [
    path('auth/registration/', RegisterView.as_view(), name='registration'),
    path('auth/login/', LoginView.as_view(), name='login'),
    path('auth/logout/', LogoutView.as_view(), name='logout'),
    path('auth/refresh/', RefreshTokenView.as_view(), name='refresh'),
    path('auth/user/', UserView.as_view(), name='user'),
    path('predict/', PredictionView.as_view(), name='predict'),
    path('user/predictions/', UserPredictionsView.as_view(), name='user-predictions'),
    path('user/predictions/<str:prediction_id>/', UserPredictionDetailView.as_view(), name='user-prediction-detail'),
    path('', include(router.urls)),
]
