import math
import torch
from pathlib import Path
import re
import datetime
from collections import Counter, defaultdict
import numpy as np
from enum import IntEnum
from diffusion_utils.experiment import DiffusionExperiment, add_exp_args
import os
from vocab import Vocab
import logging
from diffusion_utils.vae_utils import ProfileHMMSampler
from itertools import product
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import random
logger = logging.getLogger(__name__)

class nt_index(IntEnum):
    A = 0
    T = 1
    G = 2
    C = 3
    U = 1


def one_hot_index(seq):
    return [int(nt_index[char]) for char in seq]

def get_reads_with_id_prefix(path, prefix_on, prefix_off):
    reads = []
    read = ""
    switch = False
    with path.open() as f:
        for line in f.readlines():
            if line[0] == prefix_off:
                switch = False
                reads.append(read)
                read = ""
            if switch:
                read = read + line.strip()
            if line[0] == prefix_on:
                switch = True
                read = ""
        # write last read line
        reads.append(read)
    return reads

def read_fasta(path):
    return get_reads_with_id_prefix(Path(path), ">", ">")

def read_fastq(path):
    return get_reads_with_id_prefix(Path(path), "@", "+")

def read_csv(path):
    data_l = []
    raw_data = []

    with open(path, encoding='utf-8') as line_1:
        for line_2 in line_1.readlines():
            line_2 = line_2.strip("\n")
            list_1 = line_2.split(',', -1)
            raw_data.append(list_1[0])
            data_l.append([list_1[0], (float(list_1[1]), float(list_1[2]))])
        # print(data_l[0])

    return raw_data, data_l


def get_data_id(args):
    return args.dataset

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        out_data = self.data[index]
        return out_data, self.seq_len

class SingleRound:
    """pass path or raw_reads to make class of selex experiment per round.
    """

    def __init__(self, raw_reads: list = None, forward_adapter=None, reverse_adapter=None, name=None, tolerance=0, path: str = None, max_len=None, dataset=None):
        assert path is not None or raw_reads is not None, "either path or raw_reads has to be specified"
        if not os.path.exists(path): os.makedirs(path)
        if path:
            data_path = Path(path)
            if data_path.suffix == ".csv":
                logger.info("reading csv format sequence")
                raw_reads, reads = read_csv(data_path)
            elif data_path .suffix == ".fastq":
                logger.info("reading fastq format sequence")
                raw_reads = read_fastq(data_path)
            elif data_path.suffix in {".fasta", ".fa"}:
                logger.info("reading fasta format sequence")
                raw_reads = read_fasta(data_path)
            else:
                logger.critical(
                    "please specify a file with csv, fasta or fastq format")
                quit()

        self.raw_reads = raw_reads
        self.reads = reads
        self.calc_target_length()
        self.max_len = max_len

        # if forward_adapter is None or reverse_adapter is None:
        #     logger.info("adapter info not provided. estimating value")
        #     self.calc_experimental_settings()
        # else:
        #     logger.info(
        #         f"sequence design : {forward_adapter}-[random]-{reverse_adapter}")
        #     self.set_adapters(forward_adapter, reverse_adapter,
        #                       self.max_len is not None)

        if name:
            self.name = name
        else:
            self.name = re.sub(r'[-\.\:]', "",
                               str(datetime.datetime.now())).replace(" ", "_")
        logger.info(f"experiment name : {self.name}")
        self.tolerance = tolerance

        '''添加vocab相关代码'''
        # Get vocabulary
        self.vocab = Vocab(stoi={'A': 0, 'T': 1, 'G': 2, 'C': 3})


    def get_adapters(self):
        return self.forward_adapter, self.reverse_adapter

    def set_adapters(self, forward_adapter: str, reverse_adapter: str, set_max_len=False):
        self.forward_adapter = forward_adapter
        self.forward_adapter_length = len(forward_adapter)

        self.reverse_adapter = reverse_adapter
        self.reverse_adapter_length = len(reverse_adapter)

        self.random_region_length = self.target_length - \
            self.reverse_adapter_length - self.forward_adapter_length
        if set_max_len:
            self.random_region_length = self.max_len

    def calc_target_length(self):
        from collections import Counter, defaultdict
        self.read_counter = Counter(self.raw_reads)

        # calc most common length
        d = defaultdict(int)
        for key, value in self.read_counter.items():
            d[len(key)] += value
        self.target_length = sorted(d.items(), key=lambda x: -x[1])[0][0]

    def calc_experimental_settings(self):
        """calculate sequence adapters in a heuristic way
        """

        # fwd
        max_count = None
        est_adapter = ""
        for i in range(1, self.target_length):
            d = defaultdict(int)
            for seq, count in self.read_counter.most_common():
                if len(seq) < i or len(d) > 100 and seq[:i] not in d.keys():
                    continue
                d[seq[:i]] += count
            top_seq, top_count = sorted(d.items(), key=lambda x: -x[1])[0]
            if max_count is not None and top_count < max_count * 0.5:  # heuristics
                logger.info(
                    f"estimated forward adapter len is {i-1} : {est_adapter}")
                break
            max_count = sorted(d.items(), key=lambda x: -x[1])[0][1]
            if max_count < sum(self.read_counter.values()) * 0.5:
                logger.info(
                    f"no match found.")
                break
            est_adapter = top_seq
        fwd_len = i - 1
        fwd_adapter = est_adapter

        # rev
        max_count = None
        est_adapter = ""
        for i in range(1, self.target_length):
            d = defaultdict(int)
            for seq, count in self.read_counter.most_common():
                if len(seq) < i or len(d) > 100 and seq[-i:] not in d.keys():
                    continue
                d[seq[-i:]] += count
            top_seq, top_count = sorted(d.items(), key=lambda x: -x[1])[0]
            if max_count is not None and top_count < max_count * 0.5:  # heuristics
                logger.info(
                    f"estimated reverse adapter len is {i-1} : {est_adapter}")
                break
            max_count = sorted(d.items(), key=lambda x: -x[1])[0][1]
            if max_count < sum(self.read_counter.values()) * 0.5:
                logger.info(
                    f"no match found.")
                break
            est_adapter = top_seq
        rev_len = i - 1
        rev_adapter = est_adapter

        rand_len = self.target_length - rev_len - fwd_len

        logger.info(
            f"filtering with : {fwd_adapter}({fwd_len}N)-{rand_len}N-{rev_adapter}({rev_len}N)")

        # write estimated experimental settings
        self.set_adapters(fwd_adapter, rev_adapter, self.max_len is not None)

    def get_sequences_and_count(self):
        c = Counter(self.raw_reads)
        return c.most_common()

    def get_filter_passed_sequences_and_count(self, random_only=False):
        if random_only:
            return {self.cut_adapters(key): value for key, value in self.get_sequences_and_count()}
        else:
            c = Counter(self.get_filter_passed_sequences())
            return c.most_common()

    def filter_function(self, read):
        has_forward = read[: self.forward_adapter_length] == self.forward_adapter \
            or self.forward_adapter_length == 0
        has_reverse = read[-self.reverse_adapter_length:] == self.reverse_adapter \
            or self.reverse_adapter_length == 0
        match_random_region_len = abs(
            len(read) - self.target_length) <= self.tolerance
        return has_forward and has_reverse and match_random_region_len

    def get_filter_passed_sequences(self, random_only=False):
        self.filter_passed = list(filter(self.filter_function, self.raw_reads))
        if random_only:
            return [self.cut_adapters(read) for read in self.filter_passed]
        return self.filter_passed

    def cut_adapters(self, seq):
        if self.reverse_adapter_length == 0:
            ret = seq[self.forward_adapter_length:]
        else:
            ret = seq[self.forward_adapter_length: -
                      self.reverse_adapter_length]
        if self.max_len is not None:
            return ret[len(ret) // 2 - self.max_len // 2: len(ret) // 2 - self.max_len // 2 + self.max_len]
        else:
            return ret

    def __str__(self):
        return f"experiment of {len(self.raw_reads)} raw reads"

    def get_dataloader(self, min_count=1, test_size=0.1, batch_size=512, shuffle=True, use_cuda=True, num_workers=0):

        # self.min_count = min_count
        kwargs = {'num_workers': num_workers, 'pin_memory': True} if (
            use_cuda and torch.cuda.is_available()) else {}

        reads = []
        for read in self.reads:
            x = torch.LongTensor(one_hot_index(read[0]))
            z = torch.FloatTensor(read[1])
            reads.append([x, z])


        logger.info(f"# of sequences -> {len(reads)}")
        train_data, test_data = train_test_split(
            reads, test_size=test_size, shuffle=shuffle)


        train_dataset = Dataset(train_data, self.target_length)
        test_dataset = Dataset(test_data, self.target_length)

        data_shape = (self.target_length, )
        num_classes = len(self.vocab.stoi)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,  **kwargs)
        test_loader = DataLoader(
            test_dataset,  batch_size=batch_size, shuffle=False, **kwargs)
        return train_loader, test_loader, data_shape, num_classes

    def get_dataloader_reconstruct(self, min_count=1, test_size=0.1, batch_size=512, shuffle=True, use_cuda=True, num_workers=0, gmm=False):
        # self.min_count = min_count
        kwargs = {'num_workers': num_workers, 'pin_memory': True} if (
            use_cuda and torch.cuda.is_available()) else {}

        reads = []
        raw_seqs = []
        for read in self.reads:
            raw_seqs.append(read[0])
            x = torch.LongTensor(one_hot_index(read[0]))
            z = torch.FloatTensor(read[1])
            reads.append([x, z])
        logger.info(f"# of sequences -> {len(reads)}")
        dataset = Dataset(reads, self.target_length)

        data_shape = (self.target_length, )
        num_classes = len(self.vocab.stoi)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,  **kwargs)

        return raw_seqs, loader, data_shape, num_classes


class Experiment(DiffusionExperiment):

    def train_fn(self, epoch):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        loss_moving = None
        for iteration, (data, length) in enumerate(self.train_loader):
            length = length.to(self.args.device)
            x = data[0].to(self.args.device)
            mu = data[1].to(self.args.device)

            num_elem = length.sum()

            if self.args.diffusion_condition_type == 'z':
                z = mu

            loss = - self.model.log_prob(x, z).sum() / (math.log(2) * num_elem)
            loss.backward()

            if (iteration + 1) % self.args.update_freq == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler_iter: self.scheduler_iter.step()
            loss_sum += loss.detach().cpu().item() * len(x)
            loss_count += len(x)

            if loss_moving is None:
                loss_moving = loss.detach().cpu().item()
            else:
                loss_moving = .99 * loss_moving + .01 * loss.detach().cpu().item()

            if self.args.debug and loss_count > self.args.debug:
                break
            print('Training. Epoch: {}/{}, Datapoint: {}/{}, Bits/char: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.train_loader.dataset), loss_moving), end='\r')
        print('')
        if self.scheduler_epoch: self.scheduler_epoch.step()
        return {'bpc': loss_sum/loss_count}

    def eval_fn(self, epoch):
        self.model.eval()

        print('sqrt |Lt_history|^2')
        sqrt_Lt = torch.sqrt(self.model.Lt_history)
        print(' '.join(f'{item.item():.2f}' for item in sqrt_Lt))
        print()
        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            for data, length in self.eval_loader:
                length = length.to(self.args.device)
                x = data[0].to(self.args.device)
                mu = data[1].to(self.args.device)

                num_elem = length.sum()

                if self.args.diffusion_condition_type == 'z':
                    z = mu

                loss = - self.model.log_prob(x, z).sum() / (math.log(2) * num_elem)
                loss_sum += loss.detach().cpu().item() * len(x)
                loss_count += len(x)
                print('Evaluating train. Epoch: {}/{}, Datapoint: {}/{}, Bits/char: {:.3f}'.format(epoch+1, self.args.epochs, loss_count, len(self.eval_loader.dataset), loss_sum/loss_count), end='\r')
            print('')
        return {'bpc': loss_sum/loss_count}

    def get_recon_x(self, recon_params, data_shape):
        transition, emission = recon_params
        transition = transition.detach().cpu().numpy()
        emission = emission.detach().cpu().numpy()
        logger.info(f"generating sequences")
        recon_seq = []
        recon_seq_len = []
        for i, (a, e_m) in enumerate(zip(transition, emission)):
            sampler = ProfileHMMSampler(a, e_m, proba_is_log=True)
            seq_pattern = sampler.most_probable()[1].replace("_", "").replace("N", "*")
            products = product(*[list("ATGC") for _ in range(seq_pattern.count("*"))])

            rets = []
            for nt_set in products:
                ret = ""
                for part, nt in zip(seq_pattern.split("*"), list(nt_set) + [""]):
                    ret += part + nt
                rets += [ret]
            if len(rets) > self.args.eval_max:
                rets = [rets[idx] for idx in np.argsort(
                    np.random.randn(len(rets)))[:self.args.eval_max]]
            with Pool() as p:
                probas = p.map(sampler.calc_seq_proba, rets)

            most_probable_seq, min_value = sorted(
                list(zip(rets, probas)), key=lambda x: x[1])[0]
            min_value = min_value.item()
            most_probable_seq = str(most_probable_seq)
            if len(most_probable_seq) < data_shape:
                pad = ''.join(random.choice("ATGC") for i in range(data_shape - len(str(most_probable_seq))))
                most_probable_seq = most_probable_seq + pad
            if len(most_probable_seq) > data_shape:
                most_probable_seq = most_probable_seq[:int(data_shape)]
            recon_seq += [most_probable_seq]
            recon_seq_len += [len(most_probable_seq)]
        recon_x = np.array(list(map(one_hot_index, recon_seq)))
        return recon_x
