import torch
from torchsummary import summary

from models.__all_models import MiniSimpleton
import dataloader
from dataloader import PeopleDataset

from utils import print_batch_shape, setup_trainer, setup_evaluators, setup_event_handlers, setup_metrics_history, \
    plot_metrics, visualize_predictions, evaluate_model

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

    # Show batch shape
    # print_batch_shape(train_loader, "Train")
    # if valid_loader:
    #     print_batch_shape(valid_loader, "Validation")

    """Training and results"""

    LEARNING_RATE = 0.01
    MOMENTUM = 0.9

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    criterion = torch.nn.CrossEntropyLoss()

    print("Setting up trainer and evaluators...")
    trainer = setup_trainer(model, optimizer, criterion, device)
    train_evaluator, valid_evaluator = setup_evaluators(model, criterion, device)

    print("Setting up event handlers...")
    train_metrics_history, valid_metrics_history = setup_metrics_history()
    setup_event_handlers(trainer, optimizer,
                         train_evaluator, valid_evaluator,
                         train_metrics_history, valid_metrics_history,
                         train_loader, valid_loader)

    print("Training loop started")
    trainer.run(train_loader, 30)

    metrics_to_plot = ["Accuracy"]
    plot_metrics(train_metrics_history, valid_metrics_history, metrics_to_plot=metrics_to_plot)

    class_names = ['sports', 'inactivity quiet/light', 'miscellaneous', 'occupation', 'water activities',
                   'home activities', 'lawn and garden', 'religious activities', 'winter activities',
                   'conditioning exercise', 'bicycling', 'fishing and hunting', 'dancing', 'walking', 'running',
                   'self care', 'home repair', 'volunteer activities', 'music playing', 'transportation']
    visualize_predictions(model, valid_loader, device, class_names)

    # evaluate_model(model, test_loader, criterion, device)
