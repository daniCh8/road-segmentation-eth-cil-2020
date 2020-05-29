from datetime import datetime
import os
import json

config = dict()

config['net_types'] = ['u_xception', 'ures_xception', 'uspp_xception', 'u_resnet50v2',  'ures_resnet50v2', 'uspp_resnet50v2']
config['additional_epochs'] = 30
config['competition_epochs'] = 60
config['loss'] = 'dice'
config['learning_rate_additional_data'] = 1e-4
config['learning_rate_competition_data'] = 1e-5
config['treshold'] = .4

now = datetime.now()
timestamp = now.strftime("%d-%m-%Y,%H-%M")
config['model_id'] = timestamp
config['submission_root'] = '../submissions/submission_{}/'.format(config['model_id'])
config['submission_path'] = config['submission_root'] + 'submission.csv'
config['checkpoint_root'] = config['submission_root'] + 'checkpoints/'
os.makedirs(config['checkpoint_root'], exist_ok=True)

for net_type in config['net_types']:
    config[net_type] = {} #if you want to tune some specific parameter
    config[net_type]['batch_size'] = 8
    config[net_type]['checkpoint'] = config['checkpoint_root'] + '{}_checkpoint.h5'.format(net_type)

config['ures_xception']['batch_size'] = 6
    
with open(config['submission_root'] + 'config.json', 'w') as fp:
    json.dump(config, fp)