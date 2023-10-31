# De novo design and optimization of aptamers with AptaDiff


## Abstract
Aptamers are single-strand nucleic acid ligands renowned for their high affinity and specificity to target molecules. Traditionally they are identified from large DNA/RNA libraries using in vitro methods, like Systematic Evolution of Ligands by Exponential Enrichment (SELEX). However, these libraries capture only a small fraction of theoretical sequence space, and various aptamer candidates are constrained by actual sequencing capabilities from the experiment. Addressing this, we proposed AptaDiff, the first in silico aptamer design and optimization method based on the diffusion model. Our Aptadiff can generate aptamers beyond the constraints of high-throughput sequencing data, leveraging motif-dependent latent embeddings from variational autoencoder, and can optimize aptamers by affinity-guided aptamer generation according to Bayesian optimization. Comparative evaluations revealed AptaDiff’s superiority over existing aptamer generation methods in terms of quality and fidelity across four high-throughput screening data targeting distinct proteins. Moreover, our de novo designed aptamers displayed enhanced binding affinity over the top SELEX-screened experimental candidates for two target proteins. The promising results demonstrate that our AptaDiff method can significantly expedite the superior aptamer discovery.


## Tested environment

* Ubuntu == 20.04
* python == 3.8
* pytorch == 1.9.1
* cuda 11.1

## Quick Start


### Train VAE model
The initial stage is to train a VAE to learn the low-dimensional motif-dependent aptamer representation.

```
python vae/scripts/real.py   data/raw_data/datasetA_IGFBP3_P6.csv  \
                             0.001 \
```
The vae model is saved in `vae/out/trained_vae/datasetA_IGFBP3_P6_vae.mdl`

### Encode sequence to achieve latent representation
To embed the sequence, use `encode.py`, which input sequences and trained model and output sequences' representation vector. While the VAE model encodes the sequence into the latent space in the form of distribution, the output representation vector is the center of this distribution. 

Run:

```
python vae/scripts/encoder.py  data/raw_data/datasetA_IGFBP3_P6.csv \
                               vae/out/trained_vae/datasetA_IGFBP3_P6_vae.mdl                 
```

This will output sequences' representation vector in the following format:

```csv
index,seq,dim1,dim2
0,CGACATGGGCCGCCCAAGGA,0.56,0.38
1,GCGTACCGTAAATCTGTCGG,0.18,0.34
...
```
The default saving path is `vae/out/encode/embed_datasetA.csv`.

### Train Diffusion model
We convert the file `vae/out/encode/embed_datasetA.csv` into the input format of the diffusion model. The default path is `data/diffusion_data/datasetA_IGFBP3_x_z.csv`.
Run:
```
python diffusion/train.py --data_path data/diffusion_data/datasetA_IGFBP3_x_z.csv \
                          --dataset datasetA \
                          --batch_size 32 \
                          --update_freq 1 \
                          --lr 0.0001 \
                          --epochs 1000 \
                          --eval_every 2 \
                          --check_every 20 \ 
                          --diffusion_steps 1000 \ 
                          --transformer_dim 512 \
                          --transformer_heads 16 \
                          --transformer_depth 12 \
                          --transformer_blocks 1 \
                          --transformer_local_heads 8 \ 
                          --transformer_local_size 1 \
                          --gamma 0.99 \
                          --log_wandb True \
                   
```
The  aptadiff model is saved in:`diffusion/out/datasetA/aptadiff_z/.../check/checkpoint.pt`

### Run GMM
```
python vae/scripts/gmm.py  data/raw_data/datasetA_IGFBP3_P6.csv \
                           vae/out/trained_vae/datasetA_IGFBP3_P6_vae.mdl \
                           8
```
The output file is saved in: `vae/out/gmm`

### Sampling
```
python diffusion/eval_sample.py --check_path diffusion/out/datasetA/aptadiff_z/.../ \
                                --data_path data/diffusion_data/datasetA_IGFBP3_x_z.csv \
                                --eval_path vae/data/sampling_data/datasetA_IGFBP3/gmm_seq.csv \
                                --samples 8 \
                                --length 36

```
The sequence generated by sampling is saved in `results/datasetA_IGFBP3/samples/datasetA_gmm.txt`

### Run BO
```
python vae/scripts/bo.py   data/raw_data/datasetA_IGFBP3_P6.csv \
                           vae/out/trained_vae/datasetA_IGFBP3_P6_vae.mdl \
                           data/spr_data/datasetA_IGFBP3_gmm_RU \ 
                           8
```
The output file is saved in：`vae/out/bo`

Then run `diffusion/eval_sample.py` to get the BO optimized sequence
### Sampling
```
python diffusion/eval_sample.py --check_path results/datasetA_IGFBP3 \
                                --data_path data/diffusion_data/datasetA_IGFBP3_x_z.csv \
                                --eval_path vae/data/sampling_data/datasetA_IGFBP3/bo_seq.csv \
                                --samples 8 \
                                --length 36
```
The sequence generated by sampling is saved in `results/datasetA_IGFBP3/samples/datasetA_bo.txt`






