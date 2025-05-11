from Utils.DatasetProcessor import DatasetProcessor
from Utils.PointCloudTransform import PointCloudTransform
from torchvision import transforms
from Utils.DataReader import DataReading
from torch.utils.data import DataLoader, random_split
from Trainer import Trainer
import argparse
import sys
import torch
import csv
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def create_train_val_dataloaders(dataset, batch_size=6, val_split=0.2, seed=42, num_workers=0):
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


def calculate_additional_metrics(pred_point_cloud, target_point_cloud):
    if torch.is_tensor(pred_point_cloud):
        pred_np = pred_point_cloud.detach().cpu().numpy()
    else:
        pred_np = pred_point_cloud
        
    if torch.is_tensor(target_point_cloud):
        target_np = target_point_cloud.detach().cpu().numpy()
    else:
        target_np = target_point_cloud
    
    if pred_np.ndim > 2:
        pred_np = pred_np.reshape(-1, 3)
    if target_np.ndim > 2:
        target_np = target_np.reshape(-1, 3)
    
    pred_range = np.max(pred_np, axis=0) - np.min(pred_np, axis=0)
    target_range = np.max(target_np, axis=0) - np.min(target_np, axis=0)
    
    epsilon = 1e-06
    pred_range = np.where(pred_range < epsilon, epsilon, pred_range)
    target_range = np.where(target_range < epsilon, epsilon, target_range)
    
    pred_volume = np.prod(pred_range)
    target_volume = np.prod(target_range)
    
    pred_volume = max(pred_volume, epsilon)
    target_volume = max(target_volume, epsilon)
    
    pred_density = float(len(pred_np)) / float(pred_volume)
    target_density = float(len(target_np)) / float(target_volume)
    
    density_ratio = pred_density / target_density if target_density > 0 else 0
    volume_ratio = pred_volume / target_volume if target_volume > 0 else 0
    
    try:
        pred_hist, _ = np.histogramdd(pred_np, bins=10)
        target_hist, _ = np.histogramdd(target_np, bins=10)
        correlation, _ = pearsonr(pred_hist.flatten(), target_hist.flatten())
    except Exception as e:
        print(f"Warning: Could not calculate histogram correlation: {e}")
        correlation = 0
    
    return {
        'density_ratio': float(density_ratio),
        'volume_ratio': float(volume_ratio),
        'distribution_correlation': float(correlation)
    }


def plot_metrics_distribution(metrics_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    mse_values = [m['mse'] for m in metrics_list]
    mae_values = [m['mae'] for m in metrics_list]
    chamfer_values = [m['chamfer'] for m in metrics_list]
    density_ratios = [m['density_ratio'] for m in metrics_list]
    volume_ratios = [m['volume_ratio'] for m in metrics_list]
    correlations = [m['distribution_correlation'] for m in metrics_list]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Distribution of Prediction Metrics')
    
    axes[0, 0].hist(mse_values, bins=20)
    axes[0, 0].set_title('MSE Distribution')
    axes[0, 0].set_xlabel('MSE')
    axes[0, 0].set_ylabel('Frequency')
    
    axes[0, 1].hist(mae_values, bins=20)
    axes[0, 1].set_title('MAE Distribution')
    axes[0, 1].set_xlabel('MAE')
    
    axes[0, 2].hist(chamfer_values, bins=20)
    axes[0, 2].set_title('Chamfer Distance Distribution')
    axes[0, 2].set_xlabel('Chamfer Distance')
    
    axes[1, 0].hist(density_ratios, bins=20)
    axes[1, 0].set_title('Point Density Ratio Distribution')
    axes[1, 0].set_xlabel('Density Ratio')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].hist(volume_ratios, bins=20)
    axes[1, 1].set_title('Volume Ratio Distribution')
    axes[1, 1].set_xlabel('Volume Ratio')
    
    axes[1, 2].hist(correlations, bins=20)
    axes[1, 2].set_title('Distribution Correlation')
    axes[1, 2].set_xlabel('Correlation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_distribution.png'))
    plt.close()


def save_metrics_to_csv(metrics_list, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{output_file}_{timestamp}.csv"
    
    fieldnames = [
        'sample_id', 'mse', 'mae', 'chamfer_distance',
        'density_ratio', 'volume_ratio', 'distribution_correlation'
    ]
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, metrics in enumerate(metrics_list):
            row = {
                'sample_id': i,
                'mse': metrics['mse'],
                'mae': metrics['mae'],
                'chamfer_distance': metrics['chamfer'],
                'density_ratio': metrics['density_ratio'],
                'volume_ratio': metrics['volume_ratio'],
                'distribution_correlation': metrics['distribution_correlation']
            }
            writer.writerow(row)
        
        avg_metrics = {
            'sample_id': 'AVERAGE',
            'mse': np.mean([m['mse'] for m in metrics_list]),
            'mae': np.mean([m['mae'] for m in metrics_list]),
            'chamfer_distance': np.mean([m['chamfer'] for m in metrics_list]),
            'density_ratio': np.mean([m['density_ratio'] for m in metrics_list]),
            'volume_ratio': np.mean([m['volume_ratio'] for m in metrics_list]),
            'distribution_correlation': np.mean([m['distribution_correlation'] for m in metrics_list])
        }
        writer.writerow(avg_metrics)
    
    print(f"\nMetrics saved to: {filename}")
    print("\nAverage Metrics:")
    for key, value in avg_metrics.items():
        if key != 'sample_id':
            print(f"{key}: {value:.6f}")
    
    plot_metrics_distribution(metrics_list, os.path.dirname(output_file))


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
            num_epochs=100, 
            checkpoint_location="./Checkpoints", 
            verbose=verbose,
            early_stopping_patience=10,
            early_stopping_min_delta=0.001,
            learning_rate=0.00001 
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
            
            all_metrics = []

            for i, data in enumerate(train_dataloader, 0):
                left_images, left_depths, right_images, right_depths, target_point_cloud = data
                
                predicted_point_cloud, metrics = model.predict(
                    left_image=left_images, 
                    right_image=right_images,
                    target_point_cloud=target_point_cloud,
                    save_path=f"./predictions/sample_{i}"
                )
                
                if predicted_point_cloud is not None and metrics is not None:
                    additional_metrics = calculate_additional_metrics(
                        predicted_point_cloud,
                        target_point_cloud
                    )
                    metrics.update(additional_metrics)
                    
                    all_metrics.append(metrics)
                    total_mse += metrics['mse']
                    total_mae += metrics['mae']
                    total_chamfer += metrics['chamfer']
                    num_samples += 1
                    
                    print(f"\nSample {i} Metrics:")
                    print(f"MSE Loss: {metrics['mse']:.6f}")
                    print(f"MAE Loss: {metrics['mae']:.6f}")
                    print(f"Chamfer Distance: {metrics['chamfer']:.6f}")
                    print(f"Point Density Ratio: {metrics['density_ratio']:.6f}")
                    print(f"Volume Ratio: {metrics['volume_ratio']:.6f}")
                    print(f"Distribution Correlation: {metrics['distribution_correlation']:.6f}")
            
            if num_samples > 0:
                print("\nAverage Metrics Across All Samples:")
                print(f"Average MSE Loss: {total_mse / num_samples:.6f}")
                print(f"Average MAE Loss: {total_mae / num_samples:.6f}")
                print(f"Average Chamfer Distance: {total_chamfer / num_samples:.6f}")
                
                save_metrics_to_csv(all_metrics, "./metrics/prediction_metrics")
            
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