import os
from django.forms import ValidationError
import torch
import logging
from django.conf import settings
from datetime import datetime
from torchvision import transforms
import uuid
from PIL import Image
import numpy as np
from api.ai.Trainer import Trainer

logger = logging.getLogger(__name__)

def run_prediction(left_image, right_image):
    try:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        
        left_img = Image.open(left_image).convert('RGB')
        right_img = Image.open(right_image).convert('RGB')

        left_img=transform(left_img).unsqueeze(0)
        right_img=transform(right_img).unsqueeze(0)
        
        model_location = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ai/Trained_Models"))
        logger.info(f"Loading models from {model_location}")
        
        trainer = Trainer(model_location=model_location,verbose=0)
        
        trainer.load_model()
        
        point_cloud_filename = f'point_cloud_{uuid.uuid4()}.ply'
        point_cloud_path = os.path.join(settings.MEDIA_ROOT, 'point_clouds', point_cloud_filename)
        os.makedirs(os.path.dirname(point_cloud_path), exist_ok=True)
        
        logger.info("Prediction is starting")
        point_cloud, _ = trainer.predict(left_img, right_img, save_path=point_cloud_path)
        
        if point_cloud is None:
            raise ValueError("Failed to generate point cloud")
        
        logger.info(f"Point cloud shape: {point_cloud.shape}")
        logger.info(f"Point cloud type: {type(point_cloud)}")
        
        if torch.is_tensor(point_cloud):
            point_cloud_np = point_cloud.detach().cpu().numpy()
        else:
            point_cloud_np = point_cloud
            
        if point_cloud_np.ndim > 2:
            point_cloud_np = point_cloud_np.reshape(-1, 3)
            
        variance = np.var(point_cloud_np, axis=0)
        std_dev = np.std(point_cloud_np, axis=0)
        
        metrics = {
            'variance': float(np.mean(variance)),
            'std_dev': float(np.mean(std_dev)),
        }
        
        logger.info(f"Calculated metrics: {metrics}")
        
        overlay_dir = os.path.join(settings.MEDIA_ROOT, 'point_clouds')
        os.makedirs(overlay_dir, exist_ok=True)
        
        visualization_success = False
        
        actual_point_cloud_filename = f"{point_cloud_filename}_predicted.ply"
        actual_point_cloud_path = os.path.join(settings.MEDIA_ROOT, 'point_clouds', actual_point_cloud_filename)
        logger.info(f"Actual point cloud path: {actual_point_cloud_path}")
        
        relative_point_cloud_path = os.path.join('media', 'point_clouds', actual_point_cloud_filename)
        logger.info(f"Relative point cloud path: {relative_point_cloud_path}")
        
        visualization_path = None
        if visualization_success:
            visualization_path = os.path.join('media', 'point_clouds', 'visualizations', f'sample_0_overlay.png')
        
        logger.info("Starting getting unet outputs")
        left_unet, right_unet, left_depth, right_depth = trainer.getUnetOutput(left_img, right_img)
        
        unet_output_paths = {}
        depth_values = {}
        
        if left_unet is not None and right_unet is not None:
            unet_dir = os.path.join(settings.MEDIA_ROOT, 'unet_outputs')
            os.makedirs(unet_dir, exist_ok=True)
            
            if torch.is_tensor(left_unet):
                left_unet = left_unet.detach().cpu().numpy()
            if torch.is_tensor(right_unet):
                right_unet = right_unet.detach().cpu().numpy()
                
            logger.info(f"Left UNet shape before processing: {left_unet.shape}")
            logger.info(f"Right UNet shape before processing: {right_unet.shape}")
            
            left_unet = np.transpose(left_unet[0], (1, 2, 0))
            right_unet = np.transpose(right_unet[0], (1, 2, 0))
            
            left_unet = np.clip(left_unet * 255, 0, 255).astype(np.uint8)
            right_unet = np.clip(right_unet * 255, 0, 255).astype(np.uint8)
            
            logger.info(f"Left UNet shape after processing: {left_unet.shape}")
            logger.info(f"Right UNet shape after processing: {right_unet.shape}")
            
            left_unet_img = Image.fromarray(left_unet)
            right_unet_img = Image.fromarray(right_unet)
            
            left_unet_filename = f'left_unet_{uuid.uuid4()}.png'
            right_unet_filename = f'right_unet_{uuid.uuid4()}.png'
            
            left_unet_path = os.path.join(unet_dir, left_unet_filename)
            right_unet_path = os.path.join(unet_dir, right_unet_filename)
            
            left_unet_img.save(left_unet_path)
            right_unet_img.save(right_unet_path)
            
            relative_left_unet_path = os.path.join('media', 'unet_outputs', left_unet_filename)
            relative_right_unet_path = os.path.join('media', 'unet_outputs', right_unet_filename)
            
            unet_output_paths = {
                'left': relative_left_unet_path,
                'right': relative_right_unet_path
            }
            
            if left_depth is not None and right_depth is not None:
                if torch.is_tensor(left_depth):
                    left_depth = left_depth.detach().cpu().numpy()
                if torch.is_tensor(right_depth):
                    right_depth = right_depth.detach().cpu().numpy()
                
                depth_values = {
                    'left': left_depth.flatten().tolist(),
                    'right': right_depth.flatten().tolist()
                }
        
        result = {
            'point_cloud_path': relative_point_cloud_path,
            'unet_outputs': unet_output_paths,
            'depth_values': depth_values,
            'metrics': metrics
        }
        
        if visualization_path:
            result['visualization_path'] = visualization_path
            
        return result
        
    except Exception as e:
        logger.error(f"Error in run_prediction: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def save_user_images(user, left_image, right_image):
    """
    Save uploaded images with unique user ID and return their paths.
    """
    try:
        user_images_dir = os.path.join(settings.MEDIA_ROOT, 'user_images',str(user.id))
        os.makedirs(user_images_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        left_image_name = f'{user.id}_left_{timestamp}_{uuid.uuid4()}.png'
        right_image_name = f'{user.id}_right_{timestamp}_{uuid.uuid4()}.png'
        
        left_image_path = os.path.join(user_images_dir, left_image_name)
        right_image_path = os.path.join(user_images_dir, right_image_name)

        relative_left_image_path = os.path.join('media','user_images',str(user.id),left_image_name)
        relative_right_image_path = os.path.join('media','user_images',str(user.id),right_image_name)

        with open(left_image_path, 'wb') as f:
            f.write(left_image.read())
        with open(right_image_path, 'wb') as f:
            f.write(right_image.read())
        
        left_image.seek(0)
        right_image.seek(0)
        
        return left_image_path, right_image_path, relative_left_image_path, relative_right_image_path
        
    except Exception as e:
        logger.error(f"Error saving user images: {str(e)}")
        raise ValidationError("Failed to save images.")
    
def add_files_to_delete(prediction):
    file_paths = []
    if prediction.point_cloud_path:
        file_paths.append(os.path.join(settings.BASE_DIR, prediction.point_cloud_path))
    if prediction.left_unet_path:
        file_paths.append(os.path.join(settings.BASE_DIR, prediction.left_unet_path))
    if prediction.right_unet_path:
        file_paths.append(os.path.join(settings.BASE_DIR, prediction.right_unet_path))

    if 'left_image_path' in prediction.metadata:
        file_paths.append(os.path.join(settings.BASE_DIR, prediction.metadata['left_image_path']))
    if 'right_image_path' in prediction.metadata:
        file_paths.append(os.path.join(settings.BASE_DIR, prediction.metadata['right_image_path']))
    return file_paths