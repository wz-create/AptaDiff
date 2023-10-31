import logging
import click
from pathlib import Path

import torch
from torch import optim

from vae.raptgen import models
from vae.raptgen.models import CNN_PHMM_VAE, CNN_PHMM_VAE_FAST

from vae.raptgen.data import SingleRound

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
default_path = str(Path(f"{dir_path}/../out").resolve())

@click.command(help='run experiment with real data', context_settings=dict(show_default=True))
@click.argument("seqpath", type=click.Path(exists = True))
@click.option("--lr", help = "the learning rate of training", type = float)
@click.option("--epochs", help = "the number of training epochs", type = int, default = 1000)
@click.option("--threshold", help = "the number of epochs with no loss update to stop training", type = int, default = 50)
@click.option("--use-cuda/--no-cuda", help = "use cuda if available", is_flag=True, default = True)
@click.option("--cuda-id", help = "the device id of cuda to run", type = int, default = 0)
@click.option("--save-dir", help = "path to save results", type = click.Path(), default=default_path)
@click.option("--fwd", help = "forward adapter", type = str, default=None)
@click.option("--rev", help = "reverse adapter", type = str, default=None)
@click.option("--min-count", help = "minimum duplication count to pass sequence for training", type = int, default=1)
@click.option("--multi", help = "the number of training for multiple times", type = int, default=1)
@click.option("--reg-epochs", help = "the number of epochs to conduct state transition regularization", type = int, default=50)
@click.option("--embed-size", help = "the number of embedding dimension of raptgen model", type = int, default=2)
@click.option("--fast/--normal", help="[experimental] use fast calculation of probability estimation. Output of the decoder shape is different and the visualizers are not implemented.", type =bool, default= False)
def main(seqpath, lr, epochs, threshold, cuda_id, use_cuda, save_dir, fwd, rev, min_count, multi, reg_epochs, embed_size,
         fast):
    logger = logging.getLogger(__name__)

    logger.info(f"saving to {save_dir}")
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(exist_ok=True, parents=True)

    experiment = SingleRound(
        path=seqpath,
        forward_adapter=fwd,
        reverse_adapter=rev)

    # training 
    train_loader, test_loader = experiment.get_dataloader(min_count=min_count, use_cuda=use_cuda)
    device = torch.device(f"cuda:{cuda_id}" if (use_cuda and torch.cuda.is_available()) else "cpu")
    
    train_kwargs = {
        "epochs"         : epochs,
        "threshold"      : threshold,
        "device"         : device,
        "train_loader"   : train_loader,
        "test_loader"    : test_loader,
        "save_dir"       : save_dir,
        "beta_schedule"  : True, 
        "force_matching" : True,
        "force_epochs"   : reg_epochs,
    }

    # evaluate model
    target_len = experiment.random_region_length
    for i in range(multi):
        if fast:
            model = CNN_PHMM_VAE_FAST(motif_len=target_len, embed_size=embed_size)
        else:
            model = CNN_PHMM_VAE(motif_len=target_len, embed_size=embed_size)

        data_str = seqpath.split('/')[-1].split('.')[0]
        model_str = data_str
        if multi > 1:
            model_str += f"_{i}"
        model_str += "_vae.mdl"

        logger.info(f"training {model_str}")
        optimizer = optim.Adam(model.parameters(), lr)
        model = model.to(device)

        train_kwargs.update({
            "model"        : model,
            "model_str"    : model_str,
            "optimizer"    : optimizer,
            # "lr"           : lr
            })
        models.train(**train_kwargs)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    Path("./.log").mkdir(parents=True, exist_ok=True)
    formatter = '%(levelname)s : %(name)s : %(asctime)s : %(message)s'
    logging.basicConfig(
        filename='.log/logger.log',
        level=logging.DEBUG,
        format=formatter)
        
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    main()
