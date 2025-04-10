from Utils.DatasetProcessor import DatasetProcessor
from Utils.PointCloudTransform import PointCloudTransform
from torchvision import transforms
from Utils.DataReader import DataReading
from torch.utils.data import DataLoader
from Trainer import Trainer
import argparse
import sys
import os
import torch
from PIL import Image
import numpy as np


def main(args):
    # Enable verbose by default for better debugging
    verbose = args.verbose
    
    if args.generate:
        processor = DatasetProcessor(
            base_directory="E:\Dataset",
            output_base_directory="E:\Data",
            verbose=(verbose != 0),
            num_processes=4,
            excluded_folders=["dataset_4","dataset_5","dataset_6","dataset_7",]
        )
        processor.process_all_datasets()
        
    transform = transforms.Compose({
        transforms.Resize((512, 512)),
        # transforms.RandomHorizontalFlip(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ToTensor(),
    })
    transform_points = PointCloudTransform(scaling=True, normalize=True, size=512, verbose=verbose)

    try:
        dataset = DataReading(base_directory="E:\Data", transform=transform, target_transform=transform_points,
                            verbose=verbose)
        dataloader = DataLoader(dataset, batch_size=6, shuffle=False, num_workers=0)
        model = Trainer(dataloader=dataloader, num_epochs=25, checkpoint_location="./Checkpoints", verbose=verbose)
    except Exception as e:
        print(f"Error initializing dataset or dataloader: {e}")
        sys.exit(1)

    if args.train:
        try:
            print("Starting training...")
            model.train()
        except Exception as e:
            print(f"Error during training: {e}")
            sys.exit(1)

    if args.predict:
        try:
            print("Starting prediction...")
            total_mse = 0
            total_mae = 0
            total_chamfer = 0
            num_samples = 0
            model.load_model()

            for i, data in enumerate(dataloader, 0):
                left_images, left_depths, right_images, right_depths, target_point_cloud = data
                
                # Pass target_point_cloud to the predict method for comparison
                predicted_point_cloud, metrics = model.predict(
                    left_image=left_images, 
                    right_image=right_images,
                    target_point_cloud=target_point_cloud,
                    save_path=f"./predictions/sample_{i}"
                )

                left,right=model.getUnetOutput(left_images,right_images)

                unet_dir = os.path.join('./predictions', 'unet_outputs')
                os.makedirs(unet_dir, exist_ok=True)
                
                if torch.is_tensor(left):
                    left_unet = left.detach().cpu().numpy()
                if torch.is_tensor(right):
                    right_unet = right.detach().cpu().numpy()
                
                left_unet = np.transpose(left_unet[0], (1, 2, 0))
                right_unet = np.transpose(right_unet[0], (1, 2, 0))
                
                left_unet = np.clip(left_unet * 255, 0, 255).astype(np.uint8)
                right_unet = np.clip(right_unet * 255, 0, 255).astype(np.uint8)
                
                left_unet_img = Image.fromarray(left_unet)
                right_unet_img = Image.fromarray(right_unet)
                
                left_unet_path = os.path.join(unet_dir, f'left_unet.png')
                right_unet_path = os.path.join(unet_dir, f'right_unet.png')
                
                left_unet_img.save(left_unet_path)
                right_unet_img.save(right_unet_path)
                
                if metrics:
                    total_mse += metrics['mse']
                    total_mae += metrics['mae']
                    total_chamfer += metrics['chamfer']
                    num_samples += 1
                    
                    print(f"\nSample {i} Metrics:")
                    print(f"MSE Loss: {metrics['mse']:.6f}")
                    print(f"MAE Loss: {metrics['mae']:.6f}")
                    print(f"Chamfer Distance: {metrics['chamfer']:.6f}")
                
                input("Press Enter to continue to next sample...")
            
            if num_samples > 0:
                print("\nAverage Metrics Across All Samples:")
                print(f"Average MSE Loss: {total_mse / num_samples:.6f}")
                print(f"Average MAE Loss: {total_mae / num_samples:.6f}")
                print(f"Average Chamfer Distance: {total_chamfer / num_samples:.6f}")
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Selecting what should be run.")
    parser.add_argument("-g","--generate", action="store_true", help="Generate images from dataset")
    parser.add_argument("-t","--train", action="store_true", help="Train model")
    parser.add_argument("-p","--predict", action="store_true", help="Predict using model")
    parser.add_argument("-v", "--verbose", type=int, default=0, choices=[0, 1, 2], 
                      help="Verbosity level: 0=no output, 1=basic progress, 2=detailed debug")
    args = parser.parse_args()
    if not (args.generate or args.train or args.predict):
        parser.print_help()
        exit(1)
    main(args)