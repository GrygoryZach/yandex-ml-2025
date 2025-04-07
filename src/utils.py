import torch
from torch.utils.data import DataLoader
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ReduceLROnPlateauScheduler
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict
import random


def setup_engines(model, optimizer, criterion, device):
    """Создает тренажер и оценщики Ignite."""
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    valid_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    return trainer, train_evaluator, valid_evaluator


def setup_metrics_history():
    """Создает словари для хранения истории метрик."""
    train_metrics_history = defaultdict(list)
    valid_metrics_history = defaultdict(list)
    return train_metrics_history, valid_metrics_history


def log_iteration_loss(engine):
    """Выводит потери на каждой итерации."""
    print(f"Epoch[{engine.state.epoch}] - Iter[{engine.state.iteration}]: loss = {engine.state.output}")


def compute_epoch_results(train_evaluator, valid_evaluator, train_loader, valid_loader):
    """Вычисляет результаты эпохи."""
    train_evaluator.run(train_loader)
    valid_evaluator.run(valid_loader)


def log_and_save_epoch_results(engine, label, metrics_history, silent=False):
    """Выводит результаты эпохи и сохраняет метрики в историю."""
    metrics = engine.state.metrics
    metrics_items = metrics.items()
    result = ', '.join([f"{m} = {v:.4f}" for m, v in metrics_items])

    if not silent:
        print(f"{label}: {result}")

    for metric, value in metrics_items:
        metric_name = f"{label} {metric}"
        metrics_history[metric_name].append(value)


def setup_event_handlers(trainer,
                         train_evaluator, valid_evaluator,
                         optimizer,
                         train_metrics_history, valid_metrics_history,
                         train_loader, valid_loader,
                         silent=False):
    """Настраивает обработчики событий Ignite."""
    if not silent:
        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1000), log_iteration_loss)

        def log_lr():
            for param_group in optimizer.param_groups:
                print(f"Optimizer learning rate = {param_group['lr']}")
            print()

        valid_evaluator.add_event_handler(Events.COMPLETED, log_lr)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, compute_epoch_results, train_evaluator, valid_evaluator,
                              train_loader, valid_loader)
    train_evaluator.add_event_handler(Events.EPOCH_COMPLETED, log_and_save_epoch_results, label="Train",
                                      metrics_history=train_metrics_history, silent=silent)
    valid_evaluator.add_event_handler(Events.EPOCH_COMPLETED, log_and_save_epoch_results, label="Valid",
                                      metrics_history=valid_metrics_history, silent=silent)

    scheduler = ReduceLROnPlateauScheduler(optimizer, metric_name="loss", factor=0.5, patience=1, threshold=0.05)

    valid_evaluator.add_event_handler(Events.COMPLETED, scheduler)


def plot_metrics(train_metrics_history, valid_metrics_history):
    """Строит графики метрик."""
    epochs = range(1, len(train_metrics_history["Train loss"]) + 1)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_metrics_history["Train loss"], label="Train Loss", color='blue')
    plt.plot(epochs, valid_metrics_history["Valid loss"], label="Valid Loss", color='orange')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_metrics_history["Train accuracy"], label="Train Accuracy", color='blue')
    plt.plot(epochs, valid_metrics_history["Valid accuracy"], label="Valid Accuracy", color='orange')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.show()


def visualize_predictions(model, valid_loader, device, class_names, num_images=15):
    model.eval()
    images, labels = next(iter(valid_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    random_indices = random.sample(range(len(images)), num_images)
    fig, axes = plt.subplots(3, 5, figsize=(8, 5))
    axes = axes.flatten()

    for i, idx in enumerate(random_indices):
        ax = axes[i]
        ax.imshow(images[idx].cpu().numpy().transpose(1, 2, 0))

        title = f"Pred: {class_names[predicted[idx].item()]}\nTrue: {class_names[labels[idx].item()]}"
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_loader, criterion, device, out_for_table=False):
    """Оценивает модель на тестовом наборе данных после обучения."""
    metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}
    test_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    test_evaluator.run(test_loader)
    metrics = test_evaluator.state.metrics

    if out_for_table:
        params_count = sum(p.numel() for p in model.parameters())
        print(f"| {params_count} | {metrics['accuracy']:.4f} | {metrics['loss']:.4f} |")
    else:
        print(f"Test Results: Accuracy = {metrics['accuracy']:.4f}, Loss = {metrics['loss']:.4f}")
