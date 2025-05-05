import os
import sys
import logging
from rest_framework import viewsets
from rest_framework.decorators import api_view, parser_classes, permission_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from django.conf import settings
from api.prediction.models.prediction_model import PredictionResult
from api.prediction.services.prediction_service import save_user_images, run_prediction, add_files_to_delete
from api.prediction.serializers.prediction_result_serializer import PredictionResultSerializer

logger = logging.getLogger(__name__)

@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
@permission_classes([IsAuthenticated])
def predict(request):
    try:
        left_image = request.FILES.get('left_image')
        right_image = request.FILES.get('right_image')
        
        if not left_image or not right_image:
            return Response({'error': 'Both left and right images are required'}, status=400)

        if not request.user.is_authenticated:
            return Response({'error': 'Authentication required'}, status=401)
            
        logger.info(f"Received left image: {left_image.name}, right image: {right_image.name}")


        left_image_path, right_image_path, relative_left_image_path, relative_right_image_path = save_user_images(request.user, left_image, right_image)

        prediction = PredictionResult.objects.create(
            user=request.user,
            point_cloud_path='',
            left_unet_path='',
            right_unet_path='',
            metadata={
                'left_image_path': relative_left_image_path,
                'right_image_path': relative_right_image_path 
            },
            metrics={}
        )
    
        logger.info(f"Created prediction record with ID: {prediction.id}")
        
        try:
            logger.info("Starting prediction process")
            result = run_prediction(left_image_path, right_image_path)
            

            prediction.point_cloud_path = result['point_cloud_path']
            if 'visualization_path' in result:
                prediction.visualization_path = result['visualization_path']
            prediction.metrics = result.get('metrics', {})
            prediction.left_unet_path = result['unet_outputs']['left']
            prediction.right_unet_path = result['unet_outputs']['right']
            prediction.save()
            
            logger.info(f"Updated prediction record with results")
            
            return Response({
                'id': str(prediction.id),
                'point_cloud_path': prediction.point_cloud_path,
                'visualization_path': prediction.visualization_path,
                'metrics': prediction.metrics,
                'unet_outputs': result.get('unet_outputs', {}),
                'depth_values': result.get('depth_values',{}),
            })
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            prediction.metadata['error'] = str(e)
            prediction.save()
            
            return Response({
                'error': f'Error during prediction: {str(e)}',
                'id': str(prediction.id)
            }, status=500)
        
    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return Response({'error': f'Error processing images: {str(e)}'}, status=500)

class PredictionResultViewSet(viewsets.ModelViewSet):
    serializer_class = PredictionResultSerializer
    
    def get_queryset(self):
        return PredictionResult.objects.all()

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def user_prediction_history(request):
    user = request.user
    predictions = PredictionResult.objects.filter(user=user).order_by('-timestamp')
    serializer = PredictionResultSerializer(predictions, many=True)
    return Response(serializer.data)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_prediction(request, prediction_id):
    try:
        prediction = PredictionResult.objects.get(id=prediction_id, user=request.user)

        file_paths = add_files_to_delete(prediction)
        
        for file_path in file_paths:
            if os.path.exists(file_path) and settings.MEDIA_ROOT in file_path:
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete file {file_path}: {str(e)}")
        
        prediction.delete()
        return Response({'success': True})
    except PredictionResult.DoesNotExist:
        return Response({'error': 'Prediction not found'}, status=404)

@api_view(['DELETE'])
@permission_classes([IsAuthenticated])
def delete_all_user_predictions(request):
    try:
        predictions = PredictionResult.objects.filter(user=request.user)

        if not predictions:
            return Response({'message': 'No predictions found for this user.'}, status=404)

        file_paths = []
        for prediction in predictions:
            file_paths.extend(add_files_to_delete(prediction))

        for file_path in file_paths:
            if os.path.exists(file_path) and settings.MEDIA_ROOT in file_path:
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete file {file_path}: {str(e)}")

        predictions.delete()

        return Response({'success': True, 'message': 'All predictions have been deleted successfully.'})
    except Exception as e:
        logger.error(f"Error deleting predictions: {str(e)}")
        return Response({'error': 'An error occurred while deleting predictions.'}, status=500)
