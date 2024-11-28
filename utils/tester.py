import torch
import numpy as np
import wandb
from tqdm import tqdm
import time

def test(args, loader, model):
    model.eval()
    history = {}

    with torch.no_grad():
        
        for batch_idx, batch in tqdm(enumerate(loader)):
            for i in range(len(batch)):
                batch[i] = batch[i].cuda(non_blocking=True)

            # calculate VAE Loss
            logs = model.test_step(batch, compute_fid=args.compute_fid)

            # update running sums
            for k in logs.keys():
                if f'test/{k}' not in history.keys():
                    history[f'test/{k}'] = 0.
                history[f'test/{k}'] += logs[k]

        # divide by num points
        for k in history.keys():
            history[k] /= len(loader.dataset)

        # compute fid
        if args.compute_fid:
            print('Compute FID')
            history['test/fid'] = model.fid.compute()
            
        # bpd
        size_coef = args.image_size[0]*args.image_size[1]*args.image_size[2]
        bpd_coeff = 1. / np.log(2.) / size_coef
        history['test/bpd'] = history['test/nll'] * bpd_coeff

        # get random samples
        history['test/X'] = wandb.Image(batch[0][:100])
        # add reconstructions
        plot_rec = model(batch)[0]
        plot_rec = plot_rec.data.cpu()[:100]
        history['test/Recon'] = wandb.Image(plot_rec)
        # add samples
        times = []
        for temp in [1., 0.8, 0.6, 0.4]:
            start = time.time()
            sample = model.generate_x(100, t=temp)
            times.append( (time.time()-start) / 100 )
            history[f'test/Samples (t={temp})'] = wandb.Image(sample)
        history[f'test/sampling_time_per_image'] = np.array(times).mean()

        # save metrics
        print('Save Metrics')
        wandb.log(history)
