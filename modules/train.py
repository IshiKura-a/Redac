import os
from argparse import Namespace
from typing import Union

import numpy as np
import torch
from torch import nn
from torch.nn.modules import loss
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from tqdm import tqdm

from modules.loss import LossComputer, RedacLossComputer
from utils.logger import logger


def run_epoch(epoch: int, model: nn.Module, optimizer: Optimizer, loader: DataLoader,
              loss_computer: Union[LossComputer, RedacLossComputer],
              is_training: bool, show_progress: bool = False, log_every: int = 50):
    if is_training:
        model.train()
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):
            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            c = batch[3]

            outputs = model(x)
            loss_main = loss_computer.loss(outputs, y, c if is_training else g, is_training)

            if is_training:
                optimizer.zero_grad()
                loss_main.backward()
                optimizer.step()

            if is_training and (batch_idx + 1) % log_every == 0:
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()

        if (not is_training) or loss_computer.batch_count > 0:
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()


def train(model: nn.Module, criterion: nn.Module, dataset: dict, args: Namespace, epoch_offset: int):
    model = model.cuda()

    train_loss_computer = RedacLossComputer(
        criterion,
        is_robust=True,
        dataset=dataset["train_data"],
        alpha=args.alpha,
        gamma=args.gamma,
        step_size=args.robust_step_size,
    )

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)

    best_val_acc = 0
    for epoch in range(epoch_offset, epoch_offset + args.n_epochs):
        logger.info("\nEpoch [%d]:\n" % epoch)
        logger.info(f'Training:\n')
        run_epoch(
            epoch, model, optimizer,
            dataset["train_loader"],
            train_loss_computer,
            is_training=True,
            log_every=args.log_every,
        )

        logger.info(f'\nValidation:\n')
        val_loss_computer = LossComputer(
            criterion,
            is_robust=True,
            dataset=dataset["val_data"],
            step_size=args.robust_step_size,
            alpha=args.alpha)
        run_epoch(
            epoch, model, optimizer,
            dataset["val_loader"],
            val_loss_computer,
            is_training=False)

        # Test set; don't print to avoid peeking
        if dataset["test_data"] is not None:
            test_loss_computer = LossComputer(
                criterion,
                is_robust=True,
                dataset=dataset["test_data"],
                step_size=args.robust_step_size,
                alpha=args.alpha)
            run_epoch(
                epoch, model, optimizer,
                dataset["test_loader"],
                test_loss_computer,
                is_training=False)

        # Inspect learning rates
        if (epoch + 1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group["lr"]
                logger.info("Current lr: %f\n" % curr_lr)

        if epoch % args.save_step == 0:
            torch.save(model, os.path.join(args.log_dir, "%d_model.pth" % epoch))

        if args.save_last:
            torch.save(model, os.path.join(args.log_dir, "last_model.pth"))

        if args.save_best:
            if args.robust or args.reweight_groups:
                curr_val_acc = min(val_loss_computer.avg_group_acc)
            else:
                curr_val_acc = val_loss_computer.avg_acc
            logger.info(f'Current validation accuracy: {curr_val_acc}\n')
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                torch.save(model, os.path.join(args.log_dir, "best_model.pth"))
                logger.info(f'Best model saved at epoch {epoch}\n')
