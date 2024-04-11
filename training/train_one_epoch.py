import torch
from torch import nn
from typing import Iterable

from models.model import MulticlassKptsDetector
from training.loss_fn import MKDCriterion


def train_one_epoch(
        model: MulticlassKptsDetector,
        criterion: MKDCriterion,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int
):
    model.train()
    criterion.train()

    for samples, targets in data_loader:
        inputs = samples.inputs.to(device)
        classes = samples.classes.to(device)

        optimizer.zero_grad()

        outputs = model(inputs, classes)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
