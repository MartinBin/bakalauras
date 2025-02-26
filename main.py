from Utils.PhotoGenerator import PhotoGenerator
from Utils.PointCloudTransform import PointCloudTransform
from torchvision import transforms
from Utils.DataReader import DataReading
from torch.utils.data import DataLoader
from Trainer import Trainer
import argparse

def main(args):
    if args.generate:
        generator = PhotoGenerator(location="E:/Dataset/dataset_1/keyframe_1/data",save_location="./Data")
        generator.generate()

    if args.train:
        transform = transforms.Compose({
            transforms.Resize((512, 512)),
            #transforms.RandomHorizontalFlip(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.ToTensor(),
        })
        transform_points = PointCloudTransform(scaling=True,normalize=True,size=512)

        dataset = DataReading(photo_location="./Data",point_cloud_location="E:/Dataset/dataset_1/keyframe_1/", transform=transform, target_transform=transform_points)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        model = Trainer(dataloader=dataloader, num_epochs=10,checkpoint_location="./Checkpoints")
        model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Selecting what should be run.")
    parser.add_argument("-g","--generate", action="store_true", default=False, help="Generate images from dataset")
    parser.add_argument("-t","--train", action="store_true", default=False, help="Train model")
    parser.add_argument("-p","--predict", action="store_true", default=False, help="Predict using model")
    args = parser.parse_args()
    if not (args.generate or args.train or args.predict):
        parser.print_help()
        exit(1)
    main(args)