from rest_framework import serializers
from api.prediction.models.prediction_models import PredictionResult

class PredictionResultSerializer(serializers.Serializer):
    id = serializers.CharField(source='pk')
    created_at = serializers.DateTimeField()
    point_cloud_path = serializers.CharField()
    visualization_path = serializers.CharField(required=False, allow_null=True)
    metadata = serializers.DictField(required=False, default=dict)
    metrics = serializers.DictField(required=False, default=dict)
    
    def create(self, validated_data):
        return PredictionResult(**validated_data).save()
    
    def update(self, instance, validated_data):
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        return instance