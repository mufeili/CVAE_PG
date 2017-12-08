from util import Logger

import argparse
import numpy as np
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, default='Acrobot-v1_a|z(s, s_next)_bp2VAE_True')
args = parser.parse_args()

logger = Logger('/'.join(['tensorboard_log', args.directory]))

record_list = []
for folder in os.listdir(args.directory):
    record_list.append(pickle.load(open('/'.join([args.directory, folder,
                                                  'record']), 'rb')))

item_list = list(record_list[0]._fields)
record_dict_ = dict()
record_dict = dict()

for item in item_list:
    record_dict_[item] = []

for record in record_list:
    for item in item_list:
        record_dict_[item].append(np.array(getattr(record, item, None)))

for item in item_list:
    record_dict[item] = np.average(record_dict_[item], axis=0)

for _ in range(len(record_dict['cum_reward'])):
    logger.scalar_summary('cum_reward', record_dict['cum_reward'][_], _ + 1)
    logger.scalar_summary('last_hundred_average', record_dict['last_hundred_average'][_], _ + 1)
    logger.scalar_summary('policy_loss', record_dict['policy_loss'][_], _ + 1)
    logger.scalar_summary('value_loss', record_dict['value_loss'][_], _ + 1)

if 'mse_recon_loss' in item_list:
    for _ in range(len(record_dict['mse_recon_loss'])):
        logger.scalar_summary('mse_recon_loss', record_dict['mse_recon_loss'][_], _ + 1)
        logger.scalar_summary('kl_loss', record_dict['kl_loss'][_], _ + 1)
        logger.scalar_summary('vae_loss', record_dict['vae_loss'][_], _ + 1)
elif 'mse_pred_loss' in item_list:
    for _ in range(len(record_dict['mse_pred_loss'])):
        logger.scalar_summary('mse_pred_loss', record_dict['mse_pred_loss'][_], _ + 1)
        logger.scalar_summary('kl_loss', record_dict['kl_loss'][_], _ + 1)
        logger.scalar_summary('vae_loss', record_dict['vae_loss'][_], _ + 1)
