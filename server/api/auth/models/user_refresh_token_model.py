from mongoengine import Document, DateTimeField, StringField
from datetime import datetime, timedelta
from mongoengine import ReferenceField

class UserRefreshToken(Document):
    user = ReferenceField('User', required=True)
    token = StringField(required=True, unique=True)
    created_at = DateTimeField(default=datetime.utcnow)
    expires_at = DateTimeField(default=lambda: datetime.utcnow() + timedelta(days=7))