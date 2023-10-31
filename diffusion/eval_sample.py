import os
import random

import time

import numpy as np
import torch
import pickle
import argparse
import pandas as pd
from diffusion_utils.utils import add_parent_path

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

import matplotlib.font_manager as fm

# Data
add_parent_path(level=1)
from experiment import SingleRound
# Model
from model import get_model
from diffusion_utils.base import DataParallelDistribution

uniform = []

if __name__ == '__main__':
    ###########
    ## Setup ##
    ###########
    parser = argparse.ArgumentParser()
    parser.add_argument('--check_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--eval_path', type=str)
    parser.add_argument('--samples', type=int)
    parser.add_argument('--length', type=int)

    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--random_num', type=int, default=100)
    parser.add_argument('--double', type=eval, default=False)
    parser.add_argument('--benchmark', type=eval, default=False)
    eval_args = parser.parse_args()
    assert eval_args.length is not None, 'Currently, length has to be specified.'

    data_path = eval_args.data_path
    eval_path = eval_args.eval_path

    path_args = '{}/args.pickle'.format(eval_args.check_path)
    path_check = '{}/check/checkpoint.pt'.format(eval_args.check_path)

    torch.manual_seed(eval_args.seed)
    random.seed(eval_args.seed)

    ###############
    ## Load args ##
    ###############

    with open(path_args, 'rb') as f:
        args = pickle.load(f)

    ##################
    ## Specify data ##
    ##################


    experiment = SingleRound(
        path=data_path,
        dataset=args.dataset,
        forward_adapter=args.fwd,
        reverse_adapter=args.rev)
    train_loader, test_loader, data_shape, num_classes = experiment.get_dataloader(min_count=args.min_count,
                                                                                   batch_size=args.batch_size,
                                                                                   use_cuda=args.use_cuda,
                                                                                   num_workers=args.num_workers,
                                                                                   )
    ###################
    ## Specify model ##
    ###################

    model = get_model(args, data_shape=data_shape, num_classes=num_classes)
    if args.parallel == 'dp':
        model = DataParallelDistribution(model)
    checkpoint = torch.load(path_check)
    model.load_state_dict(checkpoint['model'])
    print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))

    df = pd.read_csv(eval_path)
    gmm_index, seq, x, y = df.values.T

    xy = np.column_stack((x, y))
    xy = xy.astype(np.float32)
    ############
    ## Sample ##
    ############
    path_samples = os.path.join(eval_args.check_path, 'samples/datasetA_bo.txt')
    if not os.path.exists(os.path.dirname(path_samples)):
        os.mkdir(os.path.dirname(path_samples))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model = model.eval()
    if eval_args.double:
        model = model.double()

    lengths = torch.ones(eval_args.samples, device=device, dtype=torch.long) * eval_args.length
    # mask = length_mask(lengths, maxlen=data_shape[0])

    if eval_args.benchmark:
        torch.cuda.synchronize()
        results = []
        with torch.no_grad():
            for _ in range(10):
                start = time.time()
                out = model.sample(eval_args.samples)
                torch.cuda.synchronize()
                results.append(time.time() - start)
        print()
        print(f'Sample time average {np.mean(results):.2f} +/- {np.std(results):.2f}')
        quit()

    samples_texts = []

    samples_chain, _ = model.sample_chain(args, model, eval_args.samples, xy, eval_args.length)
    samples = samples_chain[0]  # T=999
    samples_text = experiment.vocab.decode(samples.cpu(), lengths.cpu())
    samples_texts.append(samples_text)
    # print([len(s) for s in samples_text])
    with open(path_samples, 'w') as f:
        for sample_text in samples_texts:
            f.write('\n'.join(sample_text))

    def chain_linspace(samples_chain_text, num_steps=150, repeat_last=10):
        out = []
        for i in np.linspace(0, len(samples_chain_text) - 1, num_steps):
            idx = int(i)
            if idx >= len(samples_chain_text):
                print('index too big')
                idx = idx - 1
            out.append(samples_chain_text[idx])

        for i in range(repeat_last):
            out.append(samples_chain_text[-1])
        return out


    def format_text(batch_text):
        # print('batch_text', batch_text)
        out = []
        for text in batch_text:
            linesize = 90
            reformat = text[0:linesize]
            for i in range(linesize, len(text), linesize):
                reformat = reformat + '\n' + text[i:i + linesize]

            out.append(reformat)

            # print('reformat', reformat)

        return '\n\n'.join(out)


    def draw_text_to_image(text, invert_color=False):
        # font = ImageFont.truetype("CourierPrime-Regular.ttf", 24)
        font = ImageFont.truetype(fm.findfont(fm.FontProperties(family='DejaVu Sans')), 24)

        black = (0, 0, 0)
        white = (255, 255, 255)
        if invert_color:
            background_color = white
            textcolor = black
        else:
            background_color = black
            textcolor = white

        img = Image.new('RGB', (1290, 200), color=background_color)

        draw = ImageDraw.Draw(img)
        draw.multiline_text(
            (10, 10), text, textcolor, font=font)

        img_np = np.array(img)
        return img_np


    images = []
    text_chain = []


