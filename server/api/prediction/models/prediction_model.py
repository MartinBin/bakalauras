from mongoengine import Document, DateTimeField, StringField, DictField
from datetime import datetime
from mongoengine import ReferenceField

class PredictionResult(Document):
    user = ReferenceField('User', required=True)
    created_at = DateTimeField(default=datetime.utcnow)
    point_cloud_path = StringField(required=True, max_length=255)
    visualization_path = StringField(max_length=255)
    left_unet_path = StringField(max_length=255)
    right_unet_path = StringField(max_length=255)
    metadata = DictField(default=dict)
    metrics = DictField(default=dict)
    
    meta = {
        'collection': 'prediction_results',
        'ordering': ['-created_at']
    }
    
    def __str__(self):
        return f"Prediction {self.id} - {self.created_at}"