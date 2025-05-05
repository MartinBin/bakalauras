import torch
import os

checkpoint_path = './Checkpoints/best_model.pth'

model_location = 'saved_models'
os.makedirs(model_location, exist_ok=True)

best_model_state = torch.load(checkpoint_path, map_location='cpu')

torch.save(best_model_state['unet'], os.path.join(model_location, 'unet.pth'))
torch.save(best_model_state['left_encoder'], os.path.join(model_location, 'left_encoder.pth'))
torch.save(best_model_state['right_encoder'], os.path.join(model_location, 'right_encoder.pth'))
torch.save(best_model_state['decoder'], os.path.join(model_location, 'decoder.pth'))

print(f"Saved model parts to: {model_location}")
