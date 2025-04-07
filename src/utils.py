import torch
from torch.utils.data import DataLoader
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Precision, Recall, Accuracy, Fbeta, Loss
from ignite.handlers import ReduceLROnPlateauScheduler
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import defaultdict
import random
from typing import List


def setup_trainer(model, optimizer, criterion, device):
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    return trainer


def setup_evaluators(model, criterion, device):
    precision = Precision()
    recall = Recall()
    f1 = Fbeta(beta=1.0, average=False, precision=precision, recall=recall)

    metrics = {'accuracy': Accuracy(),
               'precision': precision,
               'recall': recall,
               'f1': f1,
               "loss": Loss(criterion)}

    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    valid_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device)
    return train_evaluator, valid_evaluator


def setup_metrics_history():
    train_metrics_history = defaultdict(list)
    valid_metrics_history = defaultdict(list)
    return train_metrics_history, valid_metrics_history


def log_iteration_loss(engine):
    print(f"Epoch[{engine.state.epoch}] - Iter[{engine.state.iteration}]: loss = {engine.state.output}")


def run_evaluators_on_epoch(train_evaluator, valid_evaluator, train_loader, valid_loader):
    train_evaluator.run(train_loader)
    valid_evaluator.run(valid_loader)


def log_and_save_epoch_results(engine, label, metrics_history, silent=False):
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
    if not silent:
        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=1000), log_iteration_loss)

        def log_lr():
            for param_group in optimizer.param_groups:
                print(f"Optimizer learning rate = {param_group['lr']}")
            print()

        valid_evaluator.add_event_handler(Events.COMPLETED, log_lr)

    trainer.add_event_handler(Events.EPOCH_COMPLETED,
                              run_evaluators_on_epoch,
                              train_evaluator, valid_evaluator,
                              train_loader, valid_loader)

    train_evaluator.add_event_handler(Events.EPOCH_COMPLETED,
                                      log_and_save_epoch_results,
                                      label="Train", metrics_history=train_metrics_history, silent=silent)

    valid_evaluator.add_event_handler(Events.EPOCH_COMPLETED,
                                      log_and_save_epoch_results,
                                      label="Valid", metrics_history=valid_metrics_history, silent=silent)

    scheduler = ReduceLROnPlateauScheduler(optimizer, metric_name="loss", factor=0.5, patience=1, threshold=0.05)
    valid_evaluator.add_event_handler(Events.COMPLETED, scheduler)


def plot_metrics(train_metrics_history, valid_metrics_history, metrics_to_plot: List[str]):
    epochs = range(1, len(train_metrics_history["Loss"]) + 1)
    plt.figure(figsize=(10, 5))

    def plot_metric(metric_name, train_metric_value, test_metric_value, subplot_num, epochs, is_ylim=False):
        plt.subplot(1, 2, subplot_num)
        plt.plot(epochs, train_metric_value, label=f"Train {metric_name}", color='blue')
        plt.plot(epochs, test_metric_value, label=f"Valid {metric_name}", color='orange')
        plt.title(metric_name)
        plt.xlabel("Epochs")
        plt.ylabel(metric_name)

        if is_ylim:
            plt.ylim(0, 1)

        plt.legend()
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plot_metric("Loss", train_metrics_history["Loss"], valid_metrics_history["Loss"], 1, epochs)

    for metric_name in metrics_to_plot:
        train_metric_value = train_metrics_history[metric_name]
        valid_metric_value = valid_metrics_history[metric_name]
        plot_metric(metric_name, train_metric_value, valid_metric_value, 2, epochs, is_ylim=True)

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
