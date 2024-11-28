import os
import torch
import numpy as np
import os.path as osp
import wandb
import copy
from pprint import pprint
import hydra.utils
from hydra.utils import instantiate
import omegaconf
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.distributed as dist

import utils.trainer as trainer
import utils.tester as tester
from utils.wandb import get_checkpoint
from utils.distributed_training import setup_mpi, is_main_process, cleanup

import os
os.environ[
    "HYDRA_FULL_ERROR"
] = "1"  # Makes sure that stack traces produced by hydra instantiation functions produce
# traceback errors related to the modules they built, rather than generic instantiate related errors that
# are generally useless for debugging

os.environ[
    "TORCH_DISTRIBUTED_DEBUG"
] = "DETAIL"  # extremely useful when debugging DDP setups



def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def compute_z_L_size(args):
    if args.model.name == 'ladder':
        im_size = args.dataset.image_size[1]
        hw_size = int(im_size / (2 ** len(args.model.latent_width)))
        z_L_size = (args.model.latent_width[-1], hw_size, hw_size)
    elif args.model.name == 'context_ladder':
        z_L_size = (args.dataset.image_size[0], args.model.ctx_size, args.model.ctx_size)
    if 'Diffusion' in args.model.decoder.z_L_prior._target_:
        args.model.decoder.z_L_prior.model.image_size = z_L_size[1]
        args.model.decoder.z_L_prior.model.in_channels = z_L_size[0]
        args.model.decoder.z_L_prior.model.out_channels = z_L_size[0]
        if args.model.decoder.z_L_prior.parametrization != 'eps':
            args.model.decoder.z_L_prior.model.out_channels *= 2
    else:
        args.model.decoder.z_L_prior.size = z_L_size
    return args


def init_model(args, train_loader):
    model = instantiate(args.model)
    if 'context' in args.model.name:
        model.decoder.init_dct_normalization(train_loader)

    ema_model = None
    if args.train.ema_rate > 0:
        ema_model = instantiate(args.model)
        model_params = copy.deepcopy(model.state_dict())
        ema_model.load_state_dict(model_params)
        ema_model.requires_grad_(False)
    return model, ema_model


def load_from_checkpoint(args, model, ema_model, optimizer, scheduler, scaler=None):
    if args.train.resume_id:
        chpt = get_checkpoint(args.wandb.setup,
                          idx=args.train.resume_id,
                          device='cpu'
                          )
    else:
        chpt = get_checkpoint(args.wandb.setup,
                          idx=args.train.pretrain_id,
                          device='cpu'
                          )
    args.train.start_epoch = chpt['epoch']
    if 'decoder.decoder_blocks.0.dct_scale' in chpt['model_state_dict']:
        chpt['model_state_dict']['decoder.decoder_blocks.0.dct_scale'] = model.decoder.decoder_blocks[0].dct_scale
    
    model.load_state_dict(chpt['model_state_dict'])
    if chpt['ema_model_state_dict'] is not None:
        if 'decoder.decoder_blocks.0.dct_scale' in chpt['model_state_dict']:
            chpt['ema_model_state_dict']['decoder.decoder_blocks.0.dct_scale'] = ema_model.decoder.decoder_blocks[0].dct_scale
        
        ema_model.load_state_dict(chpt['ema_model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(chpt['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(chpt['scheduler_state_dict'])
    if optimizer is not None:
        optimizer_to(optimizer, args.train.device)
    if scaler is not None:
        scaler.load_state_dict(chpt['scaler_state_dict'])
    return args, model, ema_model, optimizer, scheduler, scaler


def compute_params(model, args):
    # add network size
    vae = model.module if args.train.ddp else model
    num_param = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    enc_param = sum(p.numel() for p in vae.encoder.parameters() if p.requires_grad)
    dec_param = sum(p.numel() for p in vae.decoder.parameters() if p.requires_grad)
    prior_param = sum(
        p.numel() for p in vae.decoder.z_L_prior.parameters() if p.requires_grad
    )
    wandb.run.summary['num_parameters'] = num_param
    wandb.run.summary['encoder_parameters'] = enc_param
    wandb.run.summary['decoder_parameters'] = dec_param
    wandb.run.summary['prior_parameters'] = prior_param


@hydra.main(version_base="1.2", config_path="configs", config_name="defaults.yaml")
def run(args: omegaconf.DictConfig) -> None:
    if not args.train.ddp:
        if args.train.device[-1] == '0':
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            args.train.device = 'cuda'
        elif args.train.device[-1] == '1':
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"
            args.train.device = 'cuda'

    if args.train.ddp:
        args = setup_mpi(args)

    # infer z_L size, update the prior params
    args = compute_z_L_size(args)
    # Set the seed
    torch.manual_seed(args.train.seed)
    torch.cuda.manual_seed(args.train.seed)
    np.random.seed(args.train.seed)
    # faster matmul
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    wandb_cfg = omegaconf.OmegaConf.to_container(
        args, resolve=True, throw_on_missing=True
    )
    pprint(wandb_cfg)

    # ------------
    # data
    # ------------
    dset_params = {
        'root': os.path.join(hydra.utils.get_original_cwd(), 'datasets/')
    }
    if hasattr(args.dataset, 'overwrite_root'):
        if args.dataset.overwrite_root is not None:
            dset_params['root'] = args.dataset.overwrite_root
    if args.train.ddp:
        dset_params['ddp'] = True
        dset_params['mpi_size'] = args.mpi_size
        dset_params['rank'] = args.rank
    if 'context' in args.model.name:
        dset_params['ctx_size'] = args.model.ctx_size
    
    data_module = instantiate(args.dataset.data_module, **dset_params)
    data_module.setup('fit')
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # ------------
    # model & optimizer
    # ------------
    model, ema_model = init_model(args, train_loader)
    print(model)

    optimizer = instantiate(args.train.optimizer, params=model.parameters())
    scheduler = None
    if hasattr(args.train, "scheduler"):
        scheduler = instantiate(args.train.scheduler, optimizer=optimizer)
    if args.train.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    if args.train.resume_id is not None:
        print(f'Resume training {args.train.resume_id}')
        args, model, ema_model, optimizer, scheduler, scaler = \
            load_from_checkpoint(args, model, ema_model, optimizer, scheduler, scaler)
        model.train()
    elif args.train.pretrain_id is not None:
        print(f'Load pre-trained weights {args.train.pretrain_id}')
        args, model, ema_model, _, _, _ = \
            load_from_checkpoint(args, model, ema_model, optimizer=None, scheduler=None, scaler=None)
        args.train.start_epoch = 0
        model.train()

    if args.train.ddp:
        model = model.cuda(args.local_rank)
        model = DistributedDataParallel(model, device_ids=[args.local_rank],
                                        output_device=args.local_rank)
        if ema_model is not None:
            ema_model = ema_model.cuda(args.local_rank)
    else:
        model.to(args.train.device)
        if ema_model is not None:
            ema_model.to(args.train.device)

    # ------------
    # logging
    # ------------
    wandb.require("service")
    if is_main_process():
        tags = [
            'train_vae',
            args.dataset.name,
            args.model.name,
        ]
        if args.train.pretrain_id is not None:
            tags.append('pretrained')
        if args.train.resume_id is not None:
            wandb.init(
                **args.wandb.setup,
                id=args.train.resume_id,
                resume='must',
                settings=wandb.Settings(start_method="thread"),
            )
        else:
            wandb.init(
                **args.wandb.setup,
                config=wandb_cfg,
                group=f'{args.model.name}_{args.dataset.name}' if args.wandb.group is None else args.wandb.group,
                tags=tags,
                dir=hydra.utils.get_original_cwd(),
                settings=wandb.Settings(start_method="thread"),
                name=args.wandb.run_name,
            )
        wandb.watch(model, **args.wandb.watch)
        # define our custom x axis metric
        wandb.define_metric("epoch")
        for pref in ['train', 'val', 'z_L_prior', 'ladder_sample', 'ladder', 'misc',
                     'pic']:
            wandb.define_metric(f"{pref}/*", step_metric="epoch")
        wandb.define_metric("val/loss", summary="min", step_metric="epoch")

        # add network size
        compute_params(model, args)

    if args.train.ddp:
        dist.barrier()
    # ------------
    # training & testing
    # ------------
    # train
    if not args.train.only_test:
        trainer.train(args.train, train_loader, val_loader, model, optimizer, scheduler, ema_model, scaler)

    # save the best model
    if is_main_process():
        if osp.exists(osp.join(wandb.run.dir, 'last_chpt.pth')):
            chpt = torch.load(osp.join(wandb.run.dir, 'last_chpt.pth'))
        else:
            chpt = get_checkpoint(args.wandb.setup, idx=args.train.resume_id, device=args.train.device)
        model, ema_model = init_model(args, train_loader)
        model.load_state_dict(chpt['model_state_dict'])
        model.to(args.train.device)
        if ema_model is not None:
            ema_model.load_state_dict(chpt['ema_model_state_dict'])
            ema_model.to(args.train.device)

        # test
        data_module.setup('test')
        tester.test(args.train,
                    data_module.test_dataloader(),
                    model if ema_model is None else ema_model,
                    )
        print('Test finished')
        
        if args.train.fid_on_train:
            print('Compute FID on train')
            train_fid = model.eval_fid_on_dset(train_loader, args.train.device, temp=args.train.temp_fid)
            wandb.log({'test/train_fid': train_fid})
            
        wandb.finish()


if __name__ == "__main__":
    run()
