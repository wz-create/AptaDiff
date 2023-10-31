## run 10 motif split simulation script
import copy
import logging

import time
import click 
import numpy as np
import pandas as pd
from pathlib import Path

import torch

from vae.raptgen import models
from vae.raptgen.models import CNN_PHMM_VAE
from vae.raptgen.data import SingleRound, Result

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
default_path = str(Path(f"{dir_path}/../out/bo_{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}").resolve())


@click.command(help='run Bayesian optimization with trained model and evaluated results',
    context_settings=dict(show_default=True))
@click.argument("seqpath", type=click.Path(exists=True))
@click.argument("modelpath", type=click.Path(exists=True))
@click.argument("evalpath", type=click.Path(exists=True))
@click.option("--dim", help="num", type=int)

@click.option("--use-cuda/--no-cuda", help = "use cuda if available", is_flag=True, default = True)
@click.option("--cuda-id", help="the device id of cuda to run", type=int, default=0)
@click.option("--tag", help="output tag for input seq and model", type=str, default='')
@click.option("--fwd", help="forward adapter", type=str, default=None)
@click.option("--rev", help="reverse adapter", type=str, default=None)
@click.option("--save-dir", help="path to save results", type=click.Path(), default=default_path)
def main(seqpath, modelpath, evalpath, cuda_id, use_cuda, save_dir, tag, fwd, rev, dim):
    logger = logging.getLogger(__name__)
    save_dir = save_dir + '_' + tag if tag else save_dir
    logger.info(f"saving to {save_dir}")
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(exist_ok=True, parents=True)

    experiment = SingleRound(
        path=seqpath,
        forward_adapter=fwd,
        reverse_adapter=rev)

    target_len = experiment.random_region_length
    model = CNN_PHMM_VAE(target_len, embed_size=2)
    device = torch.device(f"cuda:{cuda_id}" if (use_cuda and torch.cuda.is_available()) else "cpu")
    model.load_state_dict(torch.load(modelpath, map_location=device))

    result = Result(
        model,
        experiment=experiment,
        path_to_save_results=save_dir,
        load_if_exists=True
    )    

    df = pd.read_csv(evalpath)
    act, seq, x, y = df.values.T

    seq_num = len(seq)
    range_length = 4
    domain = (-range_length, range_length)
    tmp = np.column_stack((x, y))
    tmp = tmp.astype(np.float32)
    result.evaluated_X = torch.from_numpy(tmp)


    result.evaluated_y = -act.astype('double')[:,None]
    # result.evaluated_y = -act[:,None]
    old_scores = result._points_to_score(copy.deepcopy(result.evaluated_X), draw_seq=True, seq_tag='origin')
    old_prob_sequences = list(zip(*old_scores))[1]

    locations = result.get_bo_result(n=dim, domain=domain, force_rerun=True)
    scores = result._points_to_score(torch.from_numpy(locations).float(), draw_seq=True,  seq_tag='bo')
    probable_sequences = list(zip(*scores))[1]
    reembed_positions = result.embed_sequences(probable_sequences)

    result.plot_bo_w_gmm(domain_range_length=range_length, extra_data=reembed_positions ,plot_range=domain, auxiliary=True, no_colors=True)
    logger.info(f"saving to {save_dir}/bo_seq.csv")
    with open(save_dir/"bo_seq.csv","w") as f:
        f.write("bo_index,seq,x,y,re_x,re_y\n")
        for i, (s, (x, y), (re_x, re_y)) in enumerate(
                zip(probable_sequences, locations, reembed_positions)):
            logger.info(f"{s},({x:.2f},{y:.2f})->({re_x:.2f},{re_y:.2f})")
            f.write(f"{i},{s},{x},{y},{re_x},{re_y}\n")

    with open(save_dir/"eval_seq.csv", "w") as f:
        f.write("eval_index,seq,x,y,re_x,re_y\n")
        for i, (s, (x, y), re_s) in enumerate(
                zip(seq, result.evaluated_X, old_prob_sequences)):
            logger.info(f"{s},({x:.2f},{y:.2f})-> {re_s}")
            f.write(f"{i},{s},{x},{y},{re_s}\n")
    if True:
    # if False:
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
