from torch import device, cuda
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torchsummary import summary

from models.__all_models import MiniSimpleton
import dataloader
from dataloader import PeopleDataset

from utils.engine import setup_trainer, setup_evaluators
from utils.logging import setup_event_handlers, setup_metrics_history
from utils.plotting import plot_metrics, visualize_predictions

if __name__ == "__main__":
    # Choose a device
    device = device("cuda" if cuda.is_available() else "cpu")
    print(device)

    # Choose a model and show its summary
    model = MiniSimpleton(device)
    summary(model, (3, 256, 512))
    print("\n")

    """Preparing the data"""

    # Setup trainers
    train_transforms = dataloader.get_train_transforms()
    val_transforms = dataloader.get_val_transforms()

    # Upload data nd initialize a dataset
    DATA_DIR = "PATH TO YOUR DATA"
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

    """Training"""

    LEARNING_RATE = 0.01
    MOMENTUM = 0.9

    optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    criterion = CrossEntropyLoss()

    print("Setting up trainer and evaluators...")
    trainer = setup_trainer(model, optimizer, criterion, device)
    train_evaluator, valid_evaluator = setup_evaluators(model, criterion, device)

    print("Setting up event handlers...")
    # Output is shown every <LOG_INTERVAL> iteration
    LOG_INTERVAL = 25
    train_metrics_history, valid_metrics_history = setup_metrics_history()
    setup_event_handlers(trainer, optimizer,
                         train_evaluator, valid_evaluator,
                         train_metrics_history, valid_metrics_history,
                         train_loader, valid_loader,
                         log_interval=LOG_INTERVAL)

    print("Training loop started\n")
    trainer.run(train_loader, 10)

    # Plot several metrics at once
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'loss']
    plot_metrics(train_metrics_history, valid_metrics_history, metrics_to_plot=metrics_to_plot)

    # To plot loss and one metric
    # plot_metric_and_loss(train_metrics_history, valid_metrics_history, "accuracy")

    class_names = ['sports', 'inactivity quiet/light', 'miscellaneous', 'occupation', 'water activities',
                   'home activities', 'lawn and garden', 'religious activities', 'winter activities',
                   'conditioning exercise', 'bicycling', 'fishing and hunting', 'dancing', 'walking', 'running',
                   'self care', 'home repair', 'volunteer activities', 'music playing', 'transportation']
    visualize_predictions(model, valid_loader, device, class_names)

    # evaluate_model(model, test_loader, criterion, device)
