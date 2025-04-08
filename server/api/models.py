from mongoengine import Document, DateTimeField, StringField, DictField
from datetime import datetime

class PredictionResult(Document):
    created_at = DateTimeField(default=datetime.utcnow)
    point_cloud_path = StringField(required=True, max_length=255)
    visualization_path = StringField(max_length=255)
    metadata = DictField(default=dict)
    
    meta = {
        'collection': 'prediction_results',
        'ordering': ['-created_at']
    }
    
    def __str__(self):
        return f"Prediction {self.id} - {self.created_at}"
