import os
import math
import time

import imageio
import numpy as np
import torch
import pickle
import argparse

from diffusion_utils.diffusion_multinomial import index_to_log_onehot
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


##########
## Setup ##
###########
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    check_path = '/home/dell/wangzhen/AptaDiff/AptaDiff1-main/diffusion/out/log/datasetA/aptadiff_z/expdecay/2023-10-30_21-49-38'
    length = 36
    data_path = '/home/dell/wangzhen/AptaDiff/AptaDiff1-main/data/diffusion_data/datasetA_IGFBP3_x_z.csv'

    parser = argparse.ArgumentParser()
    parser.add_argument('--check_path', type=str, default=check_path)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--length', type=int, default=length)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--double', type=eval, default=False)
    parser.add_argument('--benchmark', type=eval, default=False)
    eval_args = parser.parse_args()
    assert eval_args.length is not None, 'Currently, length has to be specified.'

    path_args = '{}/args.pickle'.format(eval_args.check_path)
    path_check = '{}/check/checkpoint.pt'.format(eval_args.check_path)

    torch.manual_seed(eval_args.seed)
    batch_size = eval_args.batch_size

    ###############
    ## Load args ##
    ###############
    with open(path_args, 'rb') as f:
        args = pickle.load(f)


    ##################
    ## Specify data ##
    ##################
    experiment = SingleRound(path=data_path,
                             dataset=args.dataset,
                             forward_adapter=args.fwd,
                             reverse_adapter=args.rev)

    raw_seqs, loader, data_shape, num_classes = experiment.get_dataloader_reconstruct(shuffle=False,
                                                                                      batch_size=batch_size)
    # min_count=args.min_count,
    ###################
    ## Specify model ##
    ###################

    model = get_model(args, data_shape=data_shape, num_classes=num_classes)
    if args.parallel == 'dp':
        model = DataParallelDistribution(model)
    checkpoint = torch.load(path_check)
    model.load_state_dict(checkpoint['model'])
    print('Loaded weights for model at {}/{} epochs'.format(checkpoint['current_epoch'], args.epochs))

    ############
    ## Sample ##
    ############

    path_reconstruct_output = os.path.join(eval_args.check_path, 'reconstruction/reconstruct_ep{}_s{}.txt'.
                                           format(checkpoint['current_epoch'], eval_args.seed))
    path_raw_seqs = os.path.join(eval_args.check_path, 'reconstruction/raw_seqs_ep{}_s{}.txt'.
                                 format(checkpoint['current_epoch'], eval_args.seed))
    if not os.path.exists(os.path.dirname(path_reconstruct_output)):
        os.mkdir(os.path.dirname(path_reconstruct_output))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model = model.eval()
    if eval_args.double:
        model = model.double()

    # mask = length_mask(lengths, maxlen=data_shape[0])
    # uniform_logits = torch.zeros(
    #     (eval_args.batch_size, num_classes) + data_shape, device=device)

    # log_z = log_sample_categorical(uniform_logits, 1, num_classes)
    reconstruct_chians = model.reconstruction_chain(args, model, loader, batch_size)
    reconstructs = [res[0] for res in reconstruct_chians]
    # if eval_args.benchmark:
    #     torch.cuda.synchronize()
    #     results = []
    #     with torch.no_grad():
    #         for _ in range(10):
    #             start = time.time()
    #             out = model.sample(eval_args.samples)
    #             torch.cuda.synchronize()
    #             results.append(time.time() - start)
    #     print()
    #     print(f'Sample time average {np.mean(results):.2f} +/- {np.std(results):.2f}')
    #     quit()

    # samples = reconstruct_chian[1]
    # samples = model.sample(eval_args.samples)
    # samples_text = train_loader.dataset.vocab.decode(samples.cpu(), lengths.cpu())
    reconstruct_texts = []
    lengths = torch.ones(eval_args.batch_size, device=device, dtype=torch.long) * eval_args.length
    for re in reconstructs:
        reconstruct_text = experiment.vocab.decode(re.cpu(), lengths.cpu())
        reconstruct_texts.append(reconstruct_text)
    # samples_text = experiment.vocab.decode(samples.cpu(), lengths.cpu())

    # print([len(s) for s in samples_text])
    with open(path_reconstruct_output, 'w') as f:
        for reconstruct_text in reconstruct_texts:
            f.write('\n'.join(reconstruct_text))
            f.write('\n')
    with open(path_raw_seqs, 'w') as f:
        f.write('\n'.join(raw_seqs))

    # T, B, L = reconstruct_chians[0].size()

    reconstruct_chain_texts = []
    for i, re_chain in enumerate(reconstruct_chians):
        T, B, L = reconstruct_chians[i].size()
        if B != eval_args.batch_size:
            lengths = torch.ones(B, device=device, dtype=torch.long) * eval_args.length
        reconstruct_chain_text = experiment.vocab.decode(re_chain.view(T * B, L).cpu(), lengths.repeat(T).cpu())
        reconstruct_chain_text = np.array(reconstruct_chain_text)
        reconstruct_chain_text = reconstruct_chain_text.reshape((T, B))
        reconstruct_chain_texts.append(reconstruct_chain_text)


    # samples_chain_text = experiment.vocab.decode(
    #     samples_chain.view(T * B, L).cpu(), lengths.repeat(T).cpu())
    # # print('before reshape', samples_chain_text)
    # samples_chain_text = np.array(samples_chain_text)
    # samples_chain_text = samples_chain_text.reshape((T, B))

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

    for samples_i in chain_linspace(list(reversed(reconstruct_chain_texts[0]))):
        # print('in1', samples_i)
        samples_i = format_text(samples_i)
        text_chain.append(samples_i)
        # print('in2', samples_i)
        images.append(draw_text_to_image(samples_i))

    imageio.mimsave(path_reconstruct_output[:-4] + '_chain.gif', images)

    with open(path_reconstruct_output[:-4] + '_chain.txt', 'w') as f:
        f.write('\n'.join(text_chain))
