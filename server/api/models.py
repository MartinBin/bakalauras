from mongoengine import Document, DateTimeField, StringField, DictField, EmailField, BooleanField
from datetime import datetime
from django.contrib.auth.hashers import make_password, check_password

class PredictionResult(Document):
    created_at = DateTimeField(default=datetime.utcnow)
    point_cloud_path = StringField(required=True, max_length=255)
    visualization_path = StringField(max_length=255)
    metadata = DictField(default=dict)
    metrics = DictField(default=dict)
    
    meta = {
        'collection': 'prediction_results',
        'ordering': ['-created_at']
    }
    
    def __str__(self):
        return f"Prediction {self.id} - {self.created_at}"

class User(Document):
    username = StringField(required=True, unique=True, max_length=150)
    email = EmailField(required=True, unique=True)
    password = StringField(required=True)
    is_active = BooleanField(default=True)
    is_staff = BooleanField(default=False)
    date_joined = DateTimeField(default=datetime.utcnow)
    
    meta = {
        'collection': 'users',
        'ordering': ['-date_joined']
    }
    
    def __str__(self):
        return self.username
    
    def set_password(self, raw_password):
        self.password = make_password(raw_password)
    
    def check_password(self, raw_password):
        return check_password(raw_password, self.password)
    
    @property
    def id(self):
        return str(self._id)
    
    @property
    def is_anonymous(self):
        return False
    
    @property
    def is_authenticated(self):
        return True
    
    @property
    def pk(self):
        return self.id
