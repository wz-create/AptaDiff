import logging

import click 
from pathlib import Path

import torch

from vae.raptgen.models import CNN_PHMM_VAE, CNN_PHMM_VAE_FAST
from vae.raptgen.data import SingleRound, Result
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
default_path = str(Path(f"{dir_path}/../out/encode").resolve())

@click.command(help='achieve sequence vector in embedded space.',
    context_settings=dict(show_default=True))
@click.argument("seqpath", type=click.Path(exists=True))
@click.argument("modelpath", type=click.Path(exists=True))

@click.option("--use-cuda/--no-cuda", help="use cuda if available", is_flag=True, default = True)
@click.option("--cuda-id", help="the device id of cuda to run", type=int, default = 0)
@click.option("--fwd", help="forward adapter", type = str, default='')
@click.option("--rev", help="reverse adapter", type = str, default='')
@click.option("--save-dir", help="path to save results", type = click.Path(), default=default_path)
@click.option("--fast/--normal", help="[experimental] use fast calculation of probability estimation. "
                                      "Output of the decoder shape is different and the visualizers are not implemented.", type =bool, default= False)
def main(seqpath, modelpath, cuda_id, use_cuda, save_dir, fwd, rev, fast):
    logger = logging.getLogger(__name__)
    
    logger.info(f"saving to {save_dir}")
    save_dir = Path(save_dir).expanduser()
    save_dir.mkdir(exist_ok = True, parents=True)

    experiment = SingleRound(
        path = seqpath,
        forward_adapter = fwd,
        reverse_adapter = rev)
    target_len = experiment.random_region_length
    if fast:
        model = CNN_PHMM_VAE_FAST(target_len, embed_size=2)
    else:
        model = CNN_PHMM_VAE(target_len, embed_size=2)
    device = torch.device(f"cuda:{cuda_id}" if (use_cuda and torch.cuda.is_available()) else "cpu")
    model.load_state_dict(torch.load(modelpath, map_location=device))

    result = Result(
        model,
        experiment=experiment,
        path_to_save_results=save_dir,
        load_if_exists=True
    )
    sequences = experiment.get_filter_passed_sequences(random_only=True)
    seqs = [s for s in sequences if len(s) == target_len]

    seq = set(seqs)

    embed_x = result.embed_sequences(seq)
    
    dims = embed_x.shape[1]
    logger.info(f"saving to {save_dir}/embed_datasetA.csv")
    with open(save_dir/"embed_datasetA.csv", "w") as f:
        f.write("index,seq,"+",".join([f"dim{dim+1}" for dim in range(dims)])+"\n")
        for i,(seq, X) in enumerate(zip(sequences, embed_x)):
            f.write(f"{i},{seq},"+",".join(list(map(lambda x: f"{x}", X)))+"\n")

    logger.info(f"... done.")
    
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
