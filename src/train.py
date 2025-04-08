import torch
from torchsummary import summary

from models.__all_models import MiniSimpleton
import dataloader
from dataloader import PeopleDataset

if __name__ == "__main__":
    # Choose a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Choose a model and show its summary
    model = MiniSimpleton().to(device)
    summary(model, (1, 28, 28))
    print("\n")

    # Setup trainers
    train_transforms = dataloader.get_train_transforms()
    val_transforms = dataloader.get_val_transforms()

    # Upload data nd initialize a dataset
    DATA_DIR = "D:\\yandex-ml-2025\\data\\human_poses_data"
    print("Uploading data...")
    full_dataset = PeopleDataset(DATA_DIR)

    train_set, valid_set = dataloader.split_dataset(full_dataset, valid_ratio=0.2)

    train_set.dataset.transform = train_transforms
    valid_set.dataset.transform = val_transforms

    # Setup data loaders
    print("Setting data loaders up...")
    BATCH_SIZE = 32
    train_loader, valid_loader = dataloader.setup_data_loaders(
        batch_size=BATCH_SIZE,
        train_set=train_set,
        valid_set=valid_set
    )

    train_batch = next(iter(train_loader))
    images_train, labels_train = train_batch
    print(f"Train batch shape: {images_train.shape}")

    if valid_loader:
        valid_batch = next(iter(valid_loader))
        images_valid, labels_valid = valid_batch
        print(f"Validation batch shape: {images_valid.shape}")
