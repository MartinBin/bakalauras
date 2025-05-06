from django.urls import path, include
from rest_framework.routers import DefaultRouter
from api.prediction.views.prediction_view import PredictionView, UserPredictionDetailView, UserPredictionsView, PredictionResultViewSet
from api.user.views.user_views import UserView
from api.auth.views.auth_views import LoginView, LogoutView, RefreshTokenView, RegisterView

router = DefaultRouter()
router.register(r'prediction-results', PredictionResultViewSet, basename='prediction-result')

urlpatterns = [
    path('auth/registration/', RegisterView.as_view()),
    path('auth/login/', LoginView.as_view()),
    path('auth/logout/', LogoutView.as_view()),
    path('auth/refresh/', RefreshTokenView.as_view()),
    path('auth/user/', UserView.as_view()),
    path('predict/', PredictionView.as_view()),
    path('user/predictions/', UserPredictionsView.as_view()),
    path('user/predictions/<str:prediction_id>/', UserPredictionDetailView.as_view),
    path('', include(router.urls)),
]
