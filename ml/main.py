from Utils.DatasetProcessor import DatasetProcessor
from Utils.PointCloudTransform import PointCloudTransform
from torchvision import transforms
from Utils.DataReader import DataReading
from torch.utils.data import DataLoader, random_split
from Trainer import Trainer
import argparse
import sys
import os
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d


def create_train_val_dataloaders(dataset, batch_size=6, val_split=0.2, seed=42, num_workers=0):
    """
    Split a dataset into training and validation sets and create dataloaders.
    
    This function takes a dataset and splits it into training and validation sets
    based on the specified validation split ratio. It then creates DataLoader objects
    for each set with the specified batch size and number of workers.
    
    Args:
        dataset: The dataset to split
        batch_size (int, optional): Batch size for the dataloaders. Defaults to 6.
        val_split (float, optional): Fraction of the dataset to use for validation (0.0 to 1.0). Defaults to 0.2.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 0.
        
    Returns:
        tuple: (train_dataloader, val_dataloader) - DataLoader objects for training and validation
    """
    torch.manual_seed(seed)
    
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_dataloader, val_dataloader


def main(args):
    """
    Main function to run the application based on command-line arguments.
    
    This function handles three main operations:
    1. Dataset generation: Process raw dataset files to create training data
    2. Model training: Train the 3D reconstruction model
    3. Prediction: Generate 3D point clouds from stereo images
    
    Args:
        args: Command-line arguments parsed by argparse
    """
    verbose = args.verbose
    
    if args.generate:
        processor = DatasetProcessor(
            base_directory="E:\Dataset",
            output_base_directory="E:\Data",
            verbose=(verbose != 0),
            num_processes=4,
            excluded_folders=["dataset_2","dataset_3","dataset_4","dataset_5","dataset_6","dataset_7",],
            force_regenerate=args.force_regenerate
        )
        processor.process_all_datasets()
        
    transform = transforms.Compose({
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    })
    transform_points = PointCloudTransform(scaling=True, normalize=True, size=512, verbose=verbose)

    try:
        dataset = DataReading(base_directory="E:\Data", transform=transform, target_transform=transform_points,
                            verbose=verbose)
        
        train_dataloader, val_dataloader = create_train_val_dataloaders(
            dataset, 
            batch_size=6, 
            val_split=0.2,
            seed=42,
            num_workers=0
        )
        model = Trainer(
            dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_epochs=50, 
            checkpoint_location="./Checkpoints", 
            verbose=verbose,
            early_stopping_patience=5,
            early_stopping_min_delta=0.001,
            learning_rate=0.001 
        )
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

            for i, data in enumerate(train_dataloader, 0):
                left_images, left_depths, right_images, right_depths, target_point_cloud = data
                
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
    parser.add_argument("-f", "--force-regenerate", action="store_true", 
                      help="Force regeneration of dataset files even if they already exist")
    
    args = parser.parse_args()

    if not (args.generate or args.train or args.predict):
        parser.print_help()
        exit(1)
        
    main(args)