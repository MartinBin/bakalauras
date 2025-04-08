from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import PredictionResultViewSet, predict

router = DefaultRouter()
router.register(r'results', PredictionResultViewSet)

urlpatterns = [
    path('', include(router.urls)),
    path('predict/', predict, name='predict'),
]
