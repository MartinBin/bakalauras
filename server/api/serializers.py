from rest_framework import serializers
from .models import User, PredictionResult

class UserSerializer(serializers.Serializer):
    id = serializers.CharField()
    email = serializers.EmailField()
    username = serializers.CharField()
    is_active = serializers.BooleanField()
    is_staff = serializers.BooleanField()
    date_joined = serializers.DateTimeField()

class UserRegistrationSerializer(serializers.Serializer):
    email = serializers.EmailField()
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)
    
    def create(self, validated_data):
        user = User(
            email=validated_data['email'],
            username=validated_data['username']
        )
        user.set_password(validated_data['password'])
        user.save()
        return user

class UserLoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField()

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
