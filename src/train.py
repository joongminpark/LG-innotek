import logging
import argparse
import os
import glob
import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm, trange


def train(args, train_dataset, model):
    result_writer = ResultWriter(args.eval_results_dir)

    # How to sampling
    # weighted sampling
    if args.sample_criteria == "abnormal":
        counts = train_dataset.labels
        np_counts = np.array(counts)
        # num of abnormal & normal
        num_abnormal = len(np_counts[np_counts == 1])
        num_normal = len(np_counts) - num_abnormal
        weights = [1.0 / num_abnormal if l == 1 else 1.0 / num_normal for l in counts]
        sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
    # Random sampling
    else:
        sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, sampler=sampler,)
    t_total = len(train_dataloader) * args.num_train_epochs
    args.warmup_step = int(args.warmup_percent * t_total)


    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, t_total)

    m = torch.nn.Sigmoid()
    loss_fct = torch.nn.BCELoss()

    # Train!
    logger.info("***** Running LG abnormal test *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss, logging_val_loss = 0.0, 0.0, 0.0

    best_loss = 1e10

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch",)
