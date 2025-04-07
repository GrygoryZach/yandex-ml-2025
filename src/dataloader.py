import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, List


def load_fashion_mnist(transform, data_dir='./data'):
    full_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    valid_size = len(full_dataset) - train_size

    train_set, valid_set = random_split(full_dataset, [train_size, valid_size])

    mnist_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    return train_set, valid_set, test_set, mnist_names


def display_images(dataset: Dataset, class_names: List[str], num_images=30):
    plt.figure(figsize=(10, 4))
    for i in range(min(num_images, len(dataset))):
        image, label = dataset[i]
        image = image.squeeze().numpy()
        image = (image * 255).astype('uint8')
        image = Image.fromarray(image)

        plt.subplot(3, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(class_names[label])

    plt.show()


def setup_data_loaders(batch_size: int,
                       train_set: Dataset, valid_set: Dataset, test_set: Dataset,
                       shuffle_train=True, shuffle_valid=False, shuffle_test=False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=shuffle_valid)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle_test)
    return train_loader, valid_loader, test_loader


# Example of usage
if __name__ == "__main__":
    # Define the transform to apply to the images
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the Fashion-MNIST dataset
    train_set, valid_set, test_set, mnist_names = load_fashion_mnist(transform)

    # Display the first 30 images from the training set
    display_images(train_set, mnist_names)

    # Set up the data loaders
    batch_size = 64
    train_loader, valid_loader, test_loader = setup_data_loaders(batch_size, train_set, valid_set, test_set)
