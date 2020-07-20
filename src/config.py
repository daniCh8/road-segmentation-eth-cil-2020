from datetime import datetime
import os
import json

now = datetime.now()
timestamp = now.strftime("%d-%m-%Y,%H-%M")

def create_config(net_types=[], save_dir='', batch_sizes=[], additional_epochs=[], competition_epochs=[], treshold=.4):
    config = dict()

    if net_types == []:
        net_types = ['u_xception',
                     'ures_xception',
                     'uspp_xception',
                     'u_resnet50v2',
                     'ures_resnet50v2',
                     'uspp_resnet50v2']
    config['net_types'] = net_types

    config['net_names'] = []
    for i, net_type in enumerate(net_types):
        config['net_names'].append('{}_{}'.format(str(i), net_type))

    config['loss'] = 'dice'
    config['learning_rate_additional_data'] = 1e-4
    config['learning_rate_competition_data'] = 1e-5
    config['treshold'] = treshold
    config['verbose'] = 2

    config['data_paths'] = {}
    config['data_paths']['data_dir'] = '/cluster/home/dchiappal/PochiMaPochi/data/' #root directory to change
    config['data_paths']['image_path'] = config['data_paths']['data_dir'] + 'training/images/'
    config['data_paths']['groundtruth_path'] = config['data_paths']['data_dir'] + 'training/groundtruth/'
    config['data_paths']['additional_images_path'] = config['data_paths']['data_dir'] + 'additional_data/images/'
    config['data_paths']['additional_masks_path'] = config['data_paths']['data_dir'] + 'additional_data/masks/'
    config['data_paths']['test_data_path'] = config['data_paths']['data_dir'] + 'test_images/'

    if save_dir=='':
        config['model_id'] = 'submission_{}'.format(timestamp)
    else:
        config['model_id'] = save_dir
    config['submission_root'] = '../submissions/{}/'.format(config['model_id'])
    config['submission_path'] = config['submission_root'] + 'submission.csv'
    config['figures_pdf'] = config['submission_root'] + 'predictions.pdf'
    config['checkpoint_root'] = config['submission_root'] + 'checkpoints/'
    config['prediction_root'] = config['submission_root'] + 'predictions/'
    config['csv_root'] = config['submission_root'] + 'csvs/'
    config['final_predictions_path'] = config['prediction_root'] + '{}_predictions.npy'.format('final_ensemble')
    os.makedirs(config['checkpoint_root'], exist_ok=True)
    os.makedirs(config['prediction_root'], exist_ok=True)
    os.makedirs(config['csv_root'], exist_ok=True)

    if batch_sizes == []:
        batch_sizes = [2 for i in net_types]
    else:
        assert len(batch_sizes) == len(net_types), "the number of batch sizes provided is different than the number of nets to train"
    
    if additional_epochs == []:
        additional_epochs = [40 for i in net_types]
    else:
        assert len(additional_epochs) == len(net_types), "the number of additional epochs provided is different than the number of nets to train"
    
    if competition_epochs == []:
        competition_epochs = [60 for i in net_types]
    else:
        assert len(competition_epochs) == len(net_types), "the number of competition epochs provided is different than the number of nets to train"
    
    for i, net_type in enumerate(net_types):
            net_name = config['net_names'][i]
            config[net_name] = {} 
            config[net_name]['batch_size'] = batch_sizes[i]
            config[net_name]['additional_epochs'] = additional_epochs[i]
            config[net_name]['competition_epochs'] = competition_epochs[i]
            config[net_name]['checkpoint'] = config['checkpoint_root'] + '{}_{}_{}g_{}c_weights.npy'.format(i, net_type, additional_epochs[i], competition_epochs[i])
            config[net_name]['predictions_path'] = config['prediction_root'] + '{}_{}_{}g_{}c_predictions.npy'.format(i, net_type, additional_epochs[i], competition_epochs[i])
            config[net_name]['csv_path'] = config['csv_root'] + '{}_{}_{}g_{}c_csv.csv'.format(i, net_type, additional_epochs[i], competition_epochs[i])

    files = os.listdir(config['submission_root'])
    files = [file for file in files if file.endswith('.json')]
    if len(files) != 0:
        config_name = 'config_{}.json'.format(timestamp)
    else:
        config_name = 'config.json'
    with open(config['submission_root'] + config_name, 'w') as fp:
        json.dump(config, fp, indent=4)
    
    return config


def restore_config(path=''):
    if path == '':
        path = '../submissions/submission_{}/config.json'.format(timestamp, timestamp)
    else:
        path = '../submissions/{}/config.json'.format(path, timestamp)
    with open(path, 'r') as fp:
        config = json.load(fp)
    
    return config