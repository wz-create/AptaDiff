import argparse
import sys
import os
import logging
from diffusion_utils.utils import add_parent_path, set_seeds

# Exp
from experiment import Experiment, get_data_id, SingleRound

# Data
add_parent_path(level=1)

# Model
from model import get_model, get_model_id

# Optim
from diffusion_utils.expdecay import get_optim, get_optim_id
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(sys.path)
print(os.getcwd())

###########
## Setup ##
###########
logger = logging.getLogger(__name__)
dataset_choices = {'datasetA', 'datasetB', 'datasetC', 'datasetD'}
optim_choices = {'sgd', 'adam', 'adamax'}


parser = argparse.ArgumentParser()
parser.add_argument('--debug', type=int, default=0)

'''add_exp_args'''
# Train params
parser.add_argument('--epochs', type=int)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--parallel', type=str, default=None, choices={'dp'})
parser.add_argument('--resume', type=str, default=None)
# Logging params
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--project', type=str, default=None)
parser.add_argument('--eval_every', type=int)
parser.add_argument('--check_every', type=int)
parser.add_argument('--log_tb', type=eval, default=True)
parser.add_argument('--log_wandb', type=eval)
parser.add_argument('--log_home', type=str, default='out')

'''add_data_args'''
# Data params
parser.add_argument('--dataset', type=str, choices=dataset_choices)
parser.add_argument('--data_path', type=str)
# parser.add_argument('--validation', type=eval, default=True)
parser.add_argument('--fwd', help="forward adapter", type=str, default=None)
parser.add_argument('--rev', help="reverse adapter", type=str, default=None)
parser.add_argument("--min-count", help="minimum duplication count to pass sequence for training", type=int, default=1)
# Train params
parser.add_argument('--use_cuda', type=eval, default=True)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--pin_memory', type=eval, default=False)

'''add_model_args'''
parser.add_argument('--input_dp_rate', type=float, default=0.0)
parser.add_argument('--enc_embed_size', type=int, default=2)
# Transformer params.
parser.add_argument('--transformer_dim', type=int)
parser.add_argument('--transformer_heads', type=int)
parser.add_argument('--transformer_depth', type=int)
parser.add_argument('--transformer_blocks', type=int)
parser.add_argument('--transformer_dropout', type=float, default=0.0)
parser.add_argument('--transformer_reversible', type=eval, default=False)
parser.add_argument('--transformer_local_heads', type=int)
parser.add_argument('--transformer_local_size', type=int)

parser.add_argument('--diffusion_steps', type=int)
parser.add_argument('--diffusion_sharing', type=eval, default=True)
parser.add_argument('--diffusion_loss', type=str, default='vb_stochastic')
parser.add_argument('--diffusion_parametrization', type=str, default='x0')

parser.add_argument('--diffusion_binary_scale', type=float, default=.5)


parser.add_argument('--diffusion_condition_type', type=str, default='z', choices={'z', 'x0'})
parser.add_argument('--eval-max', help="the maximum number of sequence to evaluate most probable sequence", type=int, default=256)
'''add_optim_args'''
# Model params
parser.add_argument('--optimizer', type=str, default='adam', choices=optim_choices)
parser.add_argument('--lr', type=float)
parser.add_argument('--warmup', type=int, default=None)
parser.add_argument('--update_freq', type=int)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--momentum_sqr', type=float, default=0.999)
parser.add_argument('--gamma', type=float, default=0.99)

args = parser.parse_args()
set_seeds(args.seed)

##################
## Specify data ##
##################
data_path = args.data_path
experiment = SingleRound(
        path=data_path,
        forward_adapter=args.fwd,
        reverse_adapter=args.rev,
        dataset=args.dataset)

# training
train_loader, test_loader, data_shape, num_classes = experiment.get_dataloader(
                                                      min_count=args.min_count,
                                                      batch_size=args.batch_size,
                                                      use_cuda=args.use_cuda,
                                                      num_workers=args.num_workers)

data_id = get_data_id(args)

###################
## Specify model ##
###################
# args.data_shape = data_shape[0]
model = get_model(args, data_shape=data_shape, num_classes=num_classes)
model_id = get_model_id(args)

#######################
## Specify optimizer ##
#######################

optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)
optim_id = get_optim_id(args)

##############
## Training ##
##############

exp = Experiment(args=args,
                 data_id=data_id,
                 model_id=model_id,
                 optim_id=optim_id,
                 train_loader=train_loader,
                 eval_loader=test_loader,
                 model=model,
                 optimizer=optimizer,
                 scheduler_iter=scheduler_iter,
                 scheduler_epoch=scheduler_epoch)

exp.run()
