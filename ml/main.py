from Utils.DatasetProcessor import DatasetProcessor
from Utils.PointCloudTransform import PointCloudTransform
from torchvision import transforms
from Utils.DataReader import DataReading
from torch.utils.data import DataLoader
from Trainer import Trainer
import argparse
import sys


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
                
                # Accumulate metrics if available
                if metrics:
                    total_mse += metrics['mse']
                    total_mae += metrics['mae']
                    total_chamfer += metrics['chamfer']
                    num_samples += 1
                    
                    # Print metrics for this sample
                    print(f"\nSample {i} Metrics:")
                    print(f"MSE Loss: {metrics['mse']:.6f}")
                    print(f"MAE Loss: {metrics['mae']:.6f}")
                    print(f"Chamfer Distance: {metrics['chamfer']:.6f}")
                
                # Ask for user input to continue to next sample
                input("Press Enter to continue to next sample...")
            
            # Print average metrics across all samples
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