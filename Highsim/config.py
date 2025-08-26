import torch
import numpy as np
import random
import os

# --- Model & Data Specs ---
# These parameters define the model architecture and data processing.
# They are used to generate a unique setting name for each experiment.
args = {
    'encoder_size': 64,
    'in_length': 21,
    'out_length': 20,
    'dropout': 0.05,
    'transformer_layer': 2,
    'n_head': 4,
    'gamma': 0.7,
    'f_length': 7,
    'veh_num': 6,
    'para_length': 4,
    'out_dim': 2,
    'num_mc': 0,
    'time_step': 0.1,
    'num_task': 2,
}

# --- Path Settings ---
# Automatically generate a directory name based on the hyperparameters.
setting_name = 'ed{}_in{}_out{}_drop{}_tl{}_nh{}_gamma{}'.format(
    args['encoder_size'], args['in_length'], args['out_length'],
    args['dropout'], args['transformer_layer'], args['n_head'], args['gamma']
)
# Note: Paths are set relative to the script location.
# Assumes a structure like: project_root/highsim/train.py
CHECKPOINT_PATH = os.path.join("..", "checkpoints", setting_name)
RESULT_PATH = os.path.join("..", "results", setting_name)

# Add paths to the args dictionary
args['path'] = CHECKPOINT_PATH
args['l_path'] = RESULT_PATH

# --- Device & Seed ---
SEED = 72
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
args['device'] = DEVICE

# --- Training Hyperparameters ---
args.update({
    'epoch': 20,
    'batch_size': 32,
    'learning_rate': 0.0005,
    'num_worker': 0,
})

# --- Directory Creation ---
# Automatically create directories if they don't exist.
if not os.path.exists(args['path']):
    os.makedirs(args['path'])
if not os.path.exists(args['l_path']):
    os.makedirs(args['l_path'])