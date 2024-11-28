import torch
import math
import time
import os
import numpy as np

# import datasets.dct
import wandb
import torch
# from torch.optim import Adamax, AdamW
# from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CyclicLR
import torch.nn.functional as F
import torch.distributed as dist
# from hydra.utils import instantiate

from utils.distributed_training import is_main_process


def train(args, train_loader, val_loader, model, optimizer, scheduler, ema_model=None, scaler=None):
    # compute metrics on initialization
    with torch.no_grad():
        if args.ddp:
            history_val = run_epoch(args=args,
                                    epoch=args.start_epoch,
                                    model=model.module,
                                    loader=val_loader,
                                    optimizer=None,
                                    mode='val',
                                    scaler=scaler)
        else:
            history_val = run_epoch(args=args,
                                    epoch=args.start_epoch,
                                    model=model,
                                    loader=val_loader,
                                    optimizer=None,
                                    mode='val',
                                    scaler=scaler)
        if is_main_process():
            wandb.log({**history_val, 'epoch': 0})
    best_loss = history_val['val/loss']
    e = torch.zeros(1, device=args.device)
    for epoch in range(args.start_epoch, args.max_epochs):
        if args.lr_warup_epochs > 0 and epoch < args.lr_warup_epochs:
            mult_factor = (epoch + 1) / args.lr_warup_epochs
            max_lr = args.optimizer.lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = max_lr * mult_factor
                    
        if args.ddp:
            train_loader.sampler.set_epoch(epoch)
            val_loader.sampler.set_epoch(epoch)
        time_start = time.time()
        print('Training')
        history_train = run_epoch(args,
                                  epoch=epoch,
                                  model=model.module if args.ddp else model,
                                  loader=train_loader,
                                  optimizer=optimizer,
                                  mode='train',
                                  ema_model=ema_model,
                                  scaler=scaler,
                                  max_iter=args.max_iter_per_epoch)
        train_ep_time = time.time() - time_start
        history_val = {}
        if is_main_process():
            with torch.no_grad():
                print('Validating')
                if ema_model is not None:
                    vae = ema_model
                else:
                    vae = model.module if args.ddp else model
                history_val = run_epoch(args,
                                        epoch = args.start_epoch,
                                        model=vae,
                                        loader=val_loader,
                                        optimizer=None,
                                        mode='val',
                                        scaler=scaler)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(history_val['val/loss'])
                else:
                    scheduler.step()

        hist = {**history_train, **history_val, 'train_epoch_time': train_ep_time}
        # save stats to wandb
        if is_main_process():
            wandb.log(hist)
            if hist['val/loss'] < best_loss:
                e, best_loss = torch.zeros(1, device=args.device), hist['val/loss']
                # save checkpoint
                print('->model saved<-\n')
                chpt = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if args.ddp else model.state_dict(),
                    'ema_model_state_dict': None if ema_model is None else ema_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': None if scheduler is None else scheduler.state_dict(),
                    'scaler_state_dict': None if scaler is None else scaler.state_dict(),
                    'loss': hist['val/loss'],
                }
                torch.save(chpt, os.path.join(wandb.run.dir, 'last_chpt.pth'))
                wandb.save(os.path.join(wandb.run.dir, 'last_chpt.pth'))
            else:
                e += 1
            print('Epoch: {}/{}, Time elapsed: {:.2f}s\n'
                  '* Train loss: {:.2f}  || Val.  loss: {:.2f} \n'
                  '\t Early stopping: {}/{} (BEST: {:.2f})\n'.format(
                epoch + 1, args.max_epochs, train_ep_time,
                hist['train/loss'], hist['val/loss'],
                int(e.item()), args.early_stopping_epochs, best_loss)
            )
            if math.isnan(hist['val/loss']):
                break
        
        if args.ddp:
            dist.all_reduce(e)
        # finish training if val loss is not improving anymore
        if e > args.early_stopping_epochs:
            print('->early stopping<-\n')
            break
        if not is_main_process():
            e *= 0
    
    # save the latest model weights
    print('->model saved<-\n')
    chpt = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if args.ddp else model.state_dict(),
        'ema_model_state_dict': None if ema_model is None else ema_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': None if scheduler is None else scheduler.state_dict(),
        'scaler_state_dict': None if scaler is None else scaler.state_dict(),
        'loss': hist['val/loss'],
    }
    torch.save(chpt, os.path.join(wandb.run.dir, 'last_chpt.pth'))
    wandb.save(os.path.join(wandb.run.dir, 'last_chpt.pth'))


def run_epoch(args, epoch, model, loader, optimizer, mode='train', ema_model=None, scaler=None, max_iter=None):
    if mode == 'train':
        model.train()
        model.current_epoch = epoch
        lr = optimizer.param_groups[0]["lr"]
        history = {"lr": lr, 'epoch': epoch+1, 'beta': model.get_beta()}
    elif mode == 'val':
        model.eval()
        histograms = {}
        history = {}

    for batch_idx, batch in enumerate(loader):
        if max_iter is not None and batch_idx >= max_iter:
            break
        if 'cuda' in args.device:
            for i in range(len(batch)):
                batch[i] = batch[i].cuda(non_blocking=True)
        # calculate VAE Loss
        if scaler is not None:
            with torch.autocast(device_type=args.device, dtype=torch.float16):
                loss, logs = model.train_step(batch, mode=mode)
        else:
            loss, logs = model.train_step(batch, mode=mode)

        if mode == 'train':
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            nans = torch.isnan(loss).sum()
            if batch_idx % args.acc_grad == 0:
                if nans == 0:
                    optim_step(args, model, optimizer, scaler)
                    with torch.no_grad():
                        if ema_model is not None:
                            update_ema(model, ema_model, args.ema_rate)
                optimizer.zero_grad()
            del nans

        # update running means
        N = len(loader)
        if max_iter is not None:
            N = min(N, max_iter)
        for k in logs.keys():
            h_key = k
            if '/' not in k:
                h_key = f'{mode}/{k}'
            if isinstance(logs[k], list):
                if k not in histograms.keys():
                    histograms[k] = []
                histograms[k] += logs[k]
            else:
                if h_key not in history.keys():
                    history[h_key] = 0.
                history[h_key] += logs[k] / N
        del logs

    if mode == 'val' and is_main_process():
        if (epoch % args.eval_freq) == 0:
            # add images
            history['pic/X'] = wandb.Image(batch[0][:8])
            fwd_output = model.forward(batch)
            plot_rec = fwd_output[0][:8]
            history['pic/Recon'] = wandb.Image(plot_rec)
            for t in [0.6, 0.85, 1.]:
                sample = model.generate_x(8, t=t)
                history[f'pic/Samples (t={t})'] = wandb.Image(sample)
            # add histograms
            for k in histograms.keys():
                history[f'hist/{k}'] = wandb.Histogram(torch.cat(histograms[k]))
            history.update(model.val_pics(batch, fwd_output))
    return history


def optim_step(args, model, optimizer, scaler):
    if args.grad_clip > 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e10).item()
    logs = {
        'grad_norm': grad_norm,
        'skipped_steps': 1
    }
    if args.grad_skip_thr == 0 or grad_norm < args.grad_skip_thr:
        if scaler is not None:
            scaler.step(optimizer)
        else:
            optimizer.step()
        logs['skipped_steps'] = 0

    if scaler is not None:
        scaler.update()
    if is_main_process():
        wandb.log(logs)


def update_ema(model, ema_model, ema_rate):
    for p, p_ema in zip(model.parameters(), ema_model.parameters()):
        p_ema.data.mul_(ema_rate)
        p_ema.data.add_(p.data * (1 - ema_rate))
