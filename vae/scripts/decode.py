# the position are meant to be 2 dim.
# csv file to read the target generate position should be written in the follow¥ing format
# 
# i, x, y
# 0, 0.24, 0.33
# 1, 1.0, 0.2
# ...


import logging

import click 
from pathlib import Path

import torch

from raptgen import models
from raptgen.models import CNN_PHMM_VAE
from raptgen.data import ProfileHMMSampler
from itertools import product
from multiprocessing import Pool
import pandas as pd
import numpy as np

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
default_path = str(Path(f"{dir_path}/../out/decode").resolve())

pos_path = '/home/dell/wangzhen/AI-aptamer/raptgen_our_data/out/encode/IGFBP3_embed_seq.csv'
model_path = '/home/dell/wangzhen/AI-aptamer/raptgen_our_data/data/real/datasetA_IGFBP3_P6_vae.mdl'


@click.command(help='achieve sequence vector in embedded space.',
    context_settings=dict(show_default=True))
@click.argument("pos-path", type=click.Path(exists=True), default=pos_path)
@click.argument("model-path",type=click.Path(exists=True), default=model_path)
@click.argument("target-len", type=int, default=36)
@click.option("--use-cuda/--no-cuda", help = "use cuda if available", is_flag=True, default = True)
# @click.option("--generate-seq", help = "whether or not to create most probable sequence", is_flag=True, default = True)
@click.option("--cuda-id", help = "the device id of cuda to run", type = int, default = 0)
@click.option("--save-dir", help = "path to save results", type = click.Path(), default=default_path)
@click.option("--embed-dim", help="the embedding dimension of raptgen model", type=int, default=2)
@click.option("--eval-max", help="the maximum number of sequence to evaluate most probable sequence", type=int, default=256)
def main(pos_path, model_path, target_len, cuda_id, use_cuda, save_dir, embed_dim, eval_max):
    """generate sequences from embedded sequences. Given the position of the sequence,
    the reconstructed profile HMM model is calculated.
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"saving to {save_dir}")
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(exist_ok = True, parents=True)

    device = torch.device(f"cuda:{cuda_id}" if (use_cuda and torch.cuda.is_available()) else "cpu")
    
    df = pd.read_csv(pos_path)
    arr = df.values[:, 2:].astype(float)
    arr = torch.from_numpy(arr).float().to(device)
    
    logger.info(f"loading model parameters")
    model = CNN_PHMM_VAE(target_len, embed_size=embed_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    logger.info(f"calculating phmm parameter")
    with torch.no_grad():
        model.eval()
        phmm_params = model.decoder(arr)
    
    transition, emission = phmm_params
    transition = transition.detach().cpu().numpy()
    emission = emission.detach().cpu().numpy()

    logger.info(f"generating sequences")
    scores = []
    for i, (a, e_m) in enumerate(zip(transition, emission)):
        sampler = ProfileHMMSampler(a, e_m, proba_is_log=True)
        seq_pattern = sampler.most_probable()[1].replace("_","").replace("N","*")
        products = product(*[list("ATGC") for _ in range(seq_pattern.count("*"))])

        rets = []
        for nt_set in products:
            ret = ""
            for part, nt in zip(seq_pattern.split("*"), list(nt_set)+[""]):
                ret += part+nt
            rets += [ret]
        if len(rets) > eval_max:
            rets = [rets[idx] for idx in np.argsort(
                np.random.randn(len(rets)))[:eval_max]]
        with Pool() as p:
            probas = p.map(sampler.calc_seq_proba, rets)

        most_probable_seq, min_value = sorted(
            list(zip(rets, probas)), key=lambda x: x[1])[0]
        min_value = min_value.item()
        scores += [(seq_pattern, most_probable_seq, min_value)]
        logger.info(f"{i}: {scores[-1]}")

    logger.info(f"saving to {save_dir}/IGFBP3_decode_output.csv")
    df_result = pd.DataFrame(scores, columns=["pattern","maximum_probable_sequence","log_proba"])
    df_concat = pd.concat([df,df_result],axis=1)

    df_concat.to_csv(save_dir/"IGFBP3_decode_output.csv", index=False)

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
