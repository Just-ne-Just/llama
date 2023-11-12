import torch
from torch.utils.data import DataLoader
from src.model.utils import create_mask
from timeit import default_timer as timer
from tqdm import tqdm

def train_epoch(model, optimizer, dataloader, loss_fn, pad_idx):
    model.train()
    losses = 0

    for tgt, length in tqdm(dataloader):
        tgt = tgt.to(model.device)

        tgt_input = tgt[:-1, :]

        tgt_mask, tgt_padding_mask = create_mask(tgt_input, pad_idx, model.device)

        logits = model(tgt_input, tgt_mask, tgt_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(dataloader)


def evaluate(model, dataloader, loss_fn, pad_idx):
    model.eval()
    losses = 0

    for tgt, length in tqdm(dataloader):
        src = src.to(model.device)
        tgt = tgt.to(model.device)

        tgt_input = tgt[:-1, :]

        tgt_mask, tgt_padding_mask = create_mask(tgt_input, pad_idx, model.device)

        logits = model(tgt_input, tgt_mask, tgt_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(dataloader)


def train(n_epochs, model, pad_idx, optimizer, train_loader, val_loader):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

    for epoch in range(1, n_epochs + 1):
        train_start_time = timer()
        train_loss = train_epoch(model, optimizer, train_loader, loss_fn, pad_idx)
        train_end_time = timer()
        val_loss = evaluate(model, val_loader, loss_fn, pad_idx)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(train_end_time - train_start_time):.3f}s"))