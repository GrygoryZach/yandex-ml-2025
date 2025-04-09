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
    result = ', '.join([
        f"{m} = {v.mean().item():.4f}" if isinstance(v, torch.Tensor) and v.numel() > 1 else f"{m} = {v.item():.4f}"
        if isinstance(v, torch.Tensor) else f"{m} = {v:.4f}"
        for m, v in metrics_items
    ])
    if not silent:
        print(f"{label}: {result}")

    for metric_name, value in metrics_items:
        metrics_history[metric_name].append(value)


def setup_event_handlers(trainer, optimizer,
                         train_evaluator, valid_evaluator,
                         train_metrics_history, valid_metrics_history,
                         train_loader, valid_loader,
                         silent=False, log_interval=100):
    if not silent:
        trainer.add_event_handler(Events.ITERATION_COMPLETED(every=log_interval), log_iteration_loss)

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


def print_batch_shape(data_loader: DataLoader, loader_type: str, ):
    valid_batch = next(iter(data_loader))
    images_valid, labels_valid = valid_batch
    print(f"{loader_type} batch shape: {images_valid.shape}")
