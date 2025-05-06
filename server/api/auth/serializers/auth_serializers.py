from rest_framework import serializers
from api.user.models.user_models import User

class UserLoginSerializer(serializers.Serializer):
    email = serializers.EmailField()
    password = serializers.CharField()

class UserRegistrationSerializer(serializers.Serializer):
    email = serializers.EmailField()
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)

    def validate(self, data_to_validate):
        password = data_to_validate.get('password')
        email = data_to_validate.get('email')
        username = data_to_validate.get('username')

        errors = {}

        if len(password) < 8:
            errors['password'] = "Password must be at least 8 characters long."
        
        if not any(char.isdigit() for char in password):
            errors['password'] = "Password must contain at least one digit."
        
        if not any(char.isalpha() for char in password):
            errors['password'] = "Password must contain at least one letter."

        if User.objects(email__iexact=email).count() > 0:
            errors['email'] = "Email is already in use."

        if User.objects(username__iexact=username).count() > 0:
            errors['username'] = "Username is already taken."

        if errors:
            raise serializers.ValidationError(errors)

        return data_to_validate
    
    def create(self, validated_data):
        user = User(
            email=validated_data['email'],
            username=validated_data['username']
        )
        user.set_password(validated_data['password'])
        user.save()
        return user