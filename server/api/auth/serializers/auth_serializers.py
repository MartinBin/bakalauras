from rest_framework import serializers
from api.user.models.user_models import User
import re

class UserLoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField()

class UserRegistrationSerializer(serializers.Serializer):
    email = serializers.EmailField(required=True)
    username = serializers.CharField(required=True)
    password = serializers.CharField(write_only=True, required=True)

    def validate_email(self, value):
        try:
            if User.objects.get(email__iexact=value):
                raise serializers.ValidationError("Email is already in use.")
        except User.DoesNotExist:
            pass
        return value

    def validate_username(self, value):
        try:
            if User.objects.get(username__iexact=value):
                raise serializers.ValidationError("Username is already taken.")
        except User.DoesNotExist:
            pass
        return value

    def validate_password(self, value):
        if len(value) < 8:
            raise serializers.ValidationError("Password must be at least 8 characters long.")
        if not any(char.isdigit() for char in value):
            raise serializers.ValidationError("Password must contain at least one digit.")
        if not any(char.isalpha() for char in value):
            raise serializers.ValidationError("Password must contain at least one letter.")
        return value

    def create(self, validated_data):
        user = User(
            email=validated_data['email'],
            username=validated_data['username']
        )
        user.set_password(validated_data['password'])
        user.save()
        return user