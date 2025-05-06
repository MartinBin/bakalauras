from mongoengine import Document, DateTimeField, StringField, EmailField, BooleanField
from datetime import datetime
from django.contrib.auth.hashers import make_password, check_password

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
        return str(self.id)
    
    @property
    def is_anonymous(self):
        return False
    
    @property
    def is_authenticated(self):
        return True
    
    @property
    def pk(self):
        return self.id
