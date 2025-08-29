import argparse
import os
import torch
from exp.exp_main_F import Exp_Main
import random
import numpy as np



parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer, DLinear, Linear, NLinear, SCINet, Film, SpectFlow, Real_SpectFlow, Flow_SpectFlow]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')


# DLinear
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

#Film
parser.add_argument('--ab', type=int, default=2, help='ablation version')
# SCINet

parser.add_argument('--hidden_size', default=1, type=float, help='hidden channel of module')
parser.add_argument('--kernel', default=5, type=int, help='kernel size, 3, 5, 7')
parser.add_argument('--groups', type=int, default=1)
parser.add_argument('--levels', type=int, default=3)
parser.add_argument('--stacks', type=int, default=1, help='1 stack or 2 stacks')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function, options:[mse, flow]')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

# Augmentation
parser.add_argument('--aug_method', type=str, default='NA', help='f_mask: Frequency Masking, f_mix: Frequency Mixing')
parser.add_argument('--aug_rate', type=float, default=0.5, help='mask/mix rate')
parser.add_argument('--in_batch_augmentation', action='store_true', help='Augmentation in Batch (save memory cost)', default=False)
parser.add_argument('--in_dataset_augmentation', action='store_true', help='Augmentation in Dataset', default=False)
parser.add_argument('--data_size', type=float, default=1, help='size of dataset, i.e, 0.01 represents uses 1 persent samples in the dataset')
parser.add_argument('--aug_data_size', type=int, default=1, help='size of augmented data, i.e, 1 means double the size of dataset')

parser.add_argument('--seed', type=int, default=2021, help='size of augmented data, i.e, 1 means double the size of dataset')

# continue learning
parser.add_argument('--testset_div', type=int, default=2, help='Division of dataset')
parser.add_argument('--test_time_train', type=bool, default=False, help='Affect data division')

# FLinear
parser.add_argument('--train_mode', type=int,default=0)
parser.add_argument('--cut_freq', type=int,default=0)
parser.add_argument('--base_T', type=int,default=24)
parser.add_argument('--H_order', type=int,default=2)

# Flow Matching options
parser.add_argument('--flow_on_pred_only', action='store_true', default=False, help='apply flow matching loss only on prediction horizon')
parser.add_argument('--flow_t_min', type=float, default=0.0, help='minimum time for flow sampling')
parser.add_argument('--flow_t_max', type=float, default=1.0, help='maximum time for flow sampling')
parser.add_argument('--flow_time_dim', type=int, default=16, help='time embedding dimension for flow head')
parser.add_argument('--flow_hidden_multiplier', type=float, default=2.0, help='hidden width multiplier for flow head MLP')

# Hyperparameter optimization (HPO) settings
parser.add_argument('--hpo', action='store_true', help='Enable hyperparameter optimization')
parser.add_argument('--initial_samples', type=int, default=5, help='Number of initial samples for HPO')
parser.add_argument('--num_trials', type=int, default=10, help='Number of trials for HPO')
parser.add_argument('--early_stopping', type=int, default=3, help='Patience for early stopping in HPO')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer for HPO, options: [adam, sgd, rmsprop]')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay for regularization')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.cut_freq == 0:
    args.cut_freq = int(args.seq_len // args.base_T + 1) * args.H_order + 10

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

if (not args.in_batch_augmentation) and (not args.in_dataset_augmentation):
    args.batch_size = args.batch_size * 2 

if 'noise' in args.aug_method:
    args.batch_size = args.batch_size * 2 

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

torch.cuda.empty_cache()

Exp = Exp_Main

# ------------------------------
# Optuna HPO path
# ------------------------------
if args.hpo:
    try:
        import optuna
    except Exception as e:
        raise RuntimeError("Optuna is required for --hpo. Please install it (pip install optuna).") from e

    def objective(trial: 'optuna.trial.Trial'):
        # Tune a subset of hyperparameters
        args.learning_rate = trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True)
        args.weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-2, log=True)
        args.flow_time_dim = trial.suggest_categorical('flow_time_dim', [8, 16, 32, 64])
        args.flow_hidden_multiplier = trial.suggest_float('flow_hidden_multiplier', 0.5, 8.0, log=True)
        args.beta1 = trial.suggest_float('beta1', 0.01, 0.99)
        args.beta2 = trial.suggest_float('beta2', 0.01, 0.99)
        # Prefer flow loss when using Flow_SpectFlow
        if args.model == 'Flow_SpectFlow':
            args.loss = 'flow'
        # Slightly shorter training for HPO if very large
        max_epochs = args.train_epochs
        if max_epochs > 50:
            args.train_epochs = 50

        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_H{}_trial{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.H_order,
            trial.number)

        exp = Exp(args)
        # delegate pruning via exp.train(trial=trial)
        try:
            if args.train_mode == 0:
                exp.train(setting, ft=False, trial=trial)
            elif args.train_mode == 1:
                exp.train(setting, ft=True, trial=trial)
            elif args.train_mode == 2:
                exp.train(setting, ft=False, trial=trial)
                exp.train(setting, ft=True, trial=trial)
        except optuna.TrialPruned:
            torch.cuda.empty_cache()
            raise

        # evaluate on validation
        criterion = exp._select_criterion()
        _, vali_loader = exp._get_data(flag='val')
        vali_loss = exp.vali(None, vali_loader, criterion)

        torch.cuda.empty_cache()
        return float(vali_loss)

    study = optuna.create_study(direction='minimize', storage='sqlite:///results_hpo.db', study_name='SpectFlow_hpo', load_if_exists=True)
    study.optimize(objective, n_trials=args.num_trials, n_jobs=2)
    print('Best trial:', study.best_trial.number)
    print('Best value:', study.best_value)
    print('Best params:', study.best_params)

else:
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_H{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.H_order, ii)
            args.beta1 = 0.5
            args.beta2 = 0.5
            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            if args.train_mode == 0:
                exp.train(setting, ft=False) # train on xy
            elif args.train_mode == 1:
                exp.train(setting, ft=True) # train on y
            elif args.train_mode == 2:
                exp.train(setting, ft=False)
                exp.train(setting, ft=True) # finetune

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            # if args.do_predict:
            #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            #     exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_H{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.H_order, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
