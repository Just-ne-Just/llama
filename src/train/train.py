import torch
from torch.utils.data import DataLoader
from src.model.utils import create_mask
from timeit import default_timer as timer
from tqdm.notebook import tqdm
from itertools import repeat
from src.wandb_logger.wandb import WanDBWriter


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader
        

def evaluate(model, dataloader, loss_fn, pad_idx, device):
    model.eval()
    losses = 0

    for tgt, length in tqdm(dataloader):
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]

        tgt_mask, tgt_padding_mask = create_mask(tgt_input, pad_idx, device)

        logits = model(tgt_input, tgt_mask, tgt_padding_mask)

        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(dataloader)


def train(n_epochs, model, pad_idx, optimizer, train_loader, val_loader, device, len_epoch=10000, log_step=500):
    writer = WanDBWriter()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    train_loader_inf = inf_loop(train_loader)

    for epoch in range(1, n_epochs + 1):
        losses = 0

        for i, (tgt, length) in tqdm(enumerate(tqdm(train_loader_inf, desc="train", total=len_epoch))):
            model.train()
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]

            tgt_mask, tgt_padding_mask = create_mask(tgt_input, pad_idx, device)

            logits = model(tgt_input, tgt_mask, tgt_padding_mask)

            optimizer.zero_grad()

            tgt_out = tgt[:, 1:]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            optimizer.step()
            losses += loss.item()

            if (i + 1) % len_epoch == 0:
                writer.set_step((epoch - 1) * len_epoch + i)
                val_loss = evaluate(model, val_loader, loss_fn, pad_idx, device)

                a = (i + 1) % log_step
                if a == 0:
                    a = log_step

                writer.add_scalar("train_loss", losses / a)
                writer.add_scalar("val_loss", val_loss)
                print((f"Epoch: {epoch}, Train loss: {(losses / a):.3f}, Val loss: {val_loss:.3f}"))
                break

            if (i + 1) % log_step == 0:
                writer.set_step((epoch - 1) * len_epoch + i)
                writer.add_scalar("train_loss", losses / log_step)
                print(f"Epoch: {epoch}, Train loss: {(losses / log_step):.3f}")
                losses = 0