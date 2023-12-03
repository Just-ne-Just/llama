import torch
from torch.utils.data import DataLoader
from src.model.utils import create_mask
from timeit import default_timer as timer
from tqdm.notebook import tqdm
from itertools import repeat
from src.wandb_logger.wandb import WanDBWriter
from torch import Tensor


@torch.no_grad()
def generate(model, tokenizer, batch_size: int, pad_idx, prefix: Tensor=None, max_len=384):
    """
    Samples output sequence from probability distribution obtained by model.
    if Tensor of prefix is None then full it with [BOS] token

    :params
        model: predict next token for the whole batch of sequences
        tokenizer: tokenizer for the model and [BOS] token
        batch_size: number of sequence
        prefix: Tensor of tokens with shape: [batch_size, seq_len]
        max_len: max length of predicted sequence

    :return
        the Tensor of tokens of shape: [batch_size, max_len + 1]
    """
    model.eval()
    if prefix is None:
        prefix = torch.full((batch_size, 1), fill_value=tokenizer.bos_id()).to(next(model.parameters()).device)
    
    count = max_len - prefix.shape[-1]
    for i in range(count):
        prefix = prefix.clone().detach()
        tgt_mask, tgt_padding_mask = create_mask(prefix, pad_idx, device='cuda')

        output_logits = torch.nn.functional.softmax(model.get_next_token(prefix, tgt_mask, tgt_padding_mask), dim=-1)
        
        prefix = torch.cat((prefix, torch.multinomial(output_logits, 1)), dim=-1)
    
    return prefix



def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def save_checkpoint(model, optimizer, epoch, scheduler=None):
    state = {
        "arch": type(model).__name__,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else ""
    }

    filename = "checkpoint.pth"
    torch.save(state, filename)
        

def evaluate(model, dataloader, loss_fn, pad_idx, device):
    model.eval()
    losses = 0

    for tgt, length in tqdm(dataloader):
        tgt = tgt.to(device)
        tgt_input = tgt[:, :-1]

        tgt_mask, tgt_padding_mask = create_mask(tgt_input, pad_idx, device)
        
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(tgt_input, tgt_mask, tgt_padding_mask)

        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(dataloader)


def train(n_epochs, model, pad_idx, optimizer, train_loader, val_loader, device, dataset, scheduler=None, len_epoch=10000, log_step=500):
    writer = WanDBWriter()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
    train_loader_inf = inf_loop(train_loader)

    best_loss = 1e6
    for epoch in range(1, n_epochs + 1):
        losses = 0

        for i, (tgt, length) in enumerate(tqdm(train_loader_inf, desc="train", total=len_epoch)):
            model.train()
            tgt = tgt.to(device)

            tgt_input = tgt[:, :-1]

            tgt_mask, tgt_padding_mask = create_mask(tgt_input, pad_idx, device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(tgt_input, tgt_mask, tgt_padding_mask)
                tgt_out = tgt[:, 1:]
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            loss.backward()

            optimizer.step()
            losses += loss.item()

            if (i + 1) % (len_epoch + 1) == 0:
                writer.set_step((epoch - 1) * len_epoch + i)
                val_loss = evaluate(model, val_loader, loss_fn, pad_idx, device)
                if val_loss < best_loss:
                    print("Saving checkpoint...")
                    save_checkpoint(model, optimizer, epoch, scheduler)
                    best_loss = val_loss

                a = (i + 1) % log_step
                if a == 0:
                    a = log_step

                prefix = generate(model, dataset.sp_model, 3, pad_idx)
                texts = dataset.ids2text(prefix)
                for t_num, text in enumerate(texts):
                    writer.add_text(f"step_{(epoch - 1) * len_epoch + i}_text_{t_num}", text)

                writer.add_scalar("train_loss", losses / a)
                writer.add_scalar("val_loss", val_loss)
                if scheduler is not None:
                    writer.add_scalar("lr", scheduler.get_last_lr()[0])
                print((f"Epoch: {epoch}, Train loss: {(losses / a):.3f}, Val loss: {val_loss:.3f}"))
                break

            if (i + 1) % log_step == 0:
                writer.set_step((epoch - 1) * len_epoch + i)
                if scheduler is not None:
                    writer.add_scalar("lr", scheduler.get_last_lr()[0])
                writer.add_scalar("train_loss", losses / log_step)
                print(f"Epoch: {epoch}, Train loss: {(losses / log_step):.3f}")
                losses = 0

            if scheduler is not None:
                scheduler.step()