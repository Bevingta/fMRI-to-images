import numpy as np
import pyopencl as cl
import argparse
from hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from utils import logger, local_mpi_rank, mpi_size, maybe_download, mpi_rank
from data import mkdir_p
from contextlib import contextmanager
from vae import VAE
from train_helpers import restore_params

def set_up_hyperparams(s=None):
    H = Hyperparams()
    parser = argparse.ArgumentParser()
    parser = add_vae_arguments(parser)
    parse_args_and_update_hparams(H, parser, s=s)
    logprint = logger(H.logdir)
    for i, k in enumerate(sorted(H)):
        logprint(type='hparam', key=k, value=H[k])
    np.random.seed(H.seed)
    logprint('training model', H.desc, 'on', H.dataset)
    return H, logprint

def set_up_data(H):
    shift_loss = -127.5
    scale_loss = 1. / 127.5
    
    H.image_size = 64
    H.image_channels = 3
    shift = -115.92961967
    scale = 1. / 69.37404
    
    shift_loss = np.array([shift_loss], dtype=np.float32)
    scale_loss = np.array([scale_loss], dtype=np.float32)
    
    def preprocess_func(x):
        nonlocal shift
        nonlocal scale
        nonlocal shift_loss
        nonlocal scale_loss
        inp = x.float()
        out = inp.clone()
        inp += shift
        inp *= scale
        out += shift_loss
        out *= scale_loss
        return inp, out
    
    return H, preprocess_func

def load_vaes(H, logprint=None):
    ema_vae = VAE(H)
    if H.restore_ema_path:
        print(f'Restoring ema vae from {H.restore_ema_path}')
        restore_params(ema_vae, H.restore_ema_path, map_cpu=True, local_rank=H.local_rank, mpi_size=H.mpi_size)
        ema_vae.load_state_dict(ema_vae.state_dict())
    ema_vae.requires_grad_(False)

    return ema_vae