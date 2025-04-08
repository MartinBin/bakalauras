import os
import sys
import torch
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from .models import PredictionResult
from .serializers import PredictionResultSerializer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml/")))

from Trainer import Trainer
from models.Unet import UNet
from models.Encoder import Encoder
from models.Decoder import Decoder

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def predict(request):
    try:
        # Get left and right images from request
        left_image = request.FILES.get('left_image')
        right_image = request.FILES.get('right_image')
        return_unet_outputs = request.POST.get('return_unet_outputs', 'false').lower() == 'true'
        
        if not left_image or not right_image:
            return Response({'error': 'Both left and right images are required'}, status=400)
        
        left_path = f"temp_left_{left_image.name}"
        right_path = f"temp_right_{right_image.name}"
        
        with open(left_path, 'wb+') as destination:
            for chunk in left_image.chunks():
                destination.write(chunk)
                
        with open(right_path, 'wb+') as destination:
            for chunk in right_image.chunks():
                destination.write(chunk)
        
        prediction_path, unet_outputs = run_prediction(left_path, right_path, return_unet_outputs)
        
        result = PredictionResult(
            point_cloud_path=prediction_path
        ).save()
        
        os.remove(left_path)
        os.remove(right_path)
        
        response_data = {
            'point_cloud_path': prediction_path,
            'id': str(result.id),
            'created_at': result.created_at
        }
        
        if unet_outputs:
            response_data['unet_outputs'] = unet_outputs
        
        return Response(response_data, status=status.HTTP_201_CREATED)
        
    except Exception as e:
        return Response(
            {"error": str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

class PredictionResultViewSet(viewsets.ModelViewSet):
    serializer_class = PredictionResultSerializer
    
    def get_queryset(self):
        return PredictionResult.objects.all()

def run_prediction(left_path, right_path, return_unet_outputs=False):
    from PIL import Image
    import torchvision.transforms as transforms
    import os
    import base64
    from io import BytesIO
    from django.conf import settings
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    left_image = transform(Image.open(left_path).convert('RGB')).unsqueeze(0)
    right_image = transform(Image.open(right_path).convert('RGB')).unsqueeze(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = 32
    
    trainer = Trainer(
        dataloader=None,
        num_epochs=10,
        latent_dim=latent_dim,
        checkpoint_location="./ml/Checkpoints",
        model_location="./ml/Trained_Models",
        verbose=False
    )
    
    trainer.load_model()
    
    predicted_point_cloud = trainer.predict(left_image, right_image, save_path="./predictions")
    
    unet_outputs = None
    if return_unet_outputs:
        # Get UNet outputs from the trainer
        left_unet_output = trainer.get_unet_output(left_image)
        right_unet_output = trainer.get_unet_output(right_image)
        
        # Convert tensors to images and save them
        left_unet_img = transforms.ToPILImage()(left_unet_output.squeeze(0).cpu())
        right_unet_img = transforms.ToPILImage()(right_unet_output.squeeze(0).cpu())
        
        # Create media directory if it doesn't exist
        media_predictions_dir = os.path.join(settings.MEDIA_ROOT, 'predictions')
        os.makedirs(media_predictions_dir, exist_ok=True)
        
        # Save UNet outputs to media directory
        left_unet_path = os.path.join(media_predictions_dir, "left_unet_output.png")
        right_unet_path = os.path.join(media_predictions_dir, "right_unet_output.png")
        
        left_unet_img.save(left_unet_path)
        right_unet_img.save(right_unet_path)
        
        # Create URLs for the UNet outputs
        unet_outputs = {
            'left': f"/media/predictions/left_unet_output.png",
            'right': f"/media/predictions/right_unet_output.png"
        }
    
    return "./predicted_point_cloud.ply", unet_outputs