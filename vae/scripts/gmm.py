## run 10 motif split simulation script
import logging
import time
import click 
import numpy as np
from pathlib import Path

import torch

from vae.raptgen import models
from vae.raptgen.models import CNN_PHMM_VAE
from vae.raptgen.data import SingleRound, Result

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
default_path = str(Path(f"{dir_path}/../out/gmm_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}").resolve())
torch.multiprocessing.set_sharing_strategy('file_system')

@click.command(help='select gmm center with trained model', context_settings=dict(show_default=True))

@click.argument("seqpath", type=click.Path(exists=True), default="/home/dell/wangzhen/AptaDiff/AptaDiff1-main/data/raw_data/datasetA_IGFBP3_P6.csv")
@click.argument("modelpath", type=click.Path(exists=True), default="/home/dell/wangzhen/AptaDiff/AptaDiff1-main/vae/out/trained_vae/datasetA_IGFBP3_P6_vae.mdl")
@click.option("--output_dim", help="the number of output clusters", type=int, default=8)

@click.option("--use_cuda/--no_cuda", help="use cuda if available", is_flag=True, default = True)
@click.option("--cuda_id", help="the device id of cuda to run", type=int, default=0)
@click.option("--save_dir", help="path to save results", type=click.Path(), default=default_path)
@click.option("--tag", help="output tag for input seq and model", type=str, default='')
@click.option("--fwd", help="forward adapter", type=str, default=None)
@click.option("--rev", help="reverse adapter", type=str, default=None)

def main(seqpath, modelpath, output_dim, cuda_id, use_cuda, save_dir, tag, fwd, rev):
    logger = logging.getLogger(__name__)
    save_dir = save_dir + '_' + tag if tag else save_dir
    logger.info(f"saving to {save_dir}")
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(exist_ok = True, parents=True)
    label_tag = ''
    seq_tag = 'seq'
    threshold = 10000
    experiment = SingleRound(
        path = seqpath,
        forward_adapter = fwd,
        reverse_adapter = rev,
        label_tag=label_tag,
        seq_tag=seq_tag,
    )
    target_len = experiment.random_region_length
    model = CNN_PHMM_VAE(target_len, embed_size=2)
    device = torch.device(f"cuda:{cuda_id}" if (use_cuda and torch.cuda.is_available()) else "cpu")
    model.load_state_dict(torch.load(modelpath, map_location=device))

    result = Result(
        model,
        experiment=experiment,
        path_to_save_results=save_dir,
        load_if_exists=True,
        with_label=bool(label_tag)
    )    
    sequences = result.get_gmm_probable_sequences(output_dim)
    points = result.gmm_centers
    result.plot_gmm(dim=output_dim, threshold=threshold)
    if label_tag:
        # result.plot_seqlabel(dim=output_dim, threshold=threshold)
        result.draw_seqlabel_seperated(dim=output_dim, threshold=threshold)
    
    logger.info(f"saving to {save_dir}/gmm_seq.csv")
    with open(save_dir/"gmm_seq.csv","w") as f:
        f.write("gmm_index,seq,x,y\n")
        for i,(seq,(x,y)) in enumerate(zip(sequences, points)):
            logger.info(f"{seq},({x:.2f},{y:.2f})")
            f.write(f"{i},{seq},{x},{y}\n")

    if True:
        logger.info(f"saving point num to {save_dir}/point_num.txt")
        num = result.get_points_with_filter(compare_type='less_equal', mu_filter=1)
        with open(f"{save_dir}/point_num.txt", 'w') as f:
            f.write(f"{num}")


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
