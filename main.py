from Utils.PointCloudTransform import PointCloudTransform
from torchvision import transforms
from Utils.DataReading_Old import DataReading
from torch.utils.data import DataLoader
from Trainer import Trainer

def main():
    transform = transforms.Compose({
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ToTensor(),
    })
    transform_points = PointCloudTransform(scaling=True, rotation=True)

    dataset = DataReading(root="E:/Dataset/", transform=transform, target_transform=transform_points)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = Trainer(dataloader=dataloader, num_epochs=10)
    model.train()


if __name__ == '__main__':
    main()