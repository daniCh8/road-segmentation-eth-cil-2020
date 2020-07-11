import argparse
import shutil
import os
from config import config
from model import NNet
from utils import single_model_training
import json
import numpy as np
import matplotlib.pyplot as plt

def main_train(config):
    for net_to_train in config['net_types']:
        net = NNet(net_type=net_to_train, data_paths=config['data_paths'])
        single_model_training(model=net,
                              save_path=config[net_to_train]['checkpoint'],
                              additional_epochs=config['additional_epochs'],
                              competition_epochs=config['competition_epochs'],
                              b_size=config[net_to_train]['batch_size'],
                              loss=config['loss'],
                              l_rate_a=config['learning_rate_additional_data'],
                              l_rate_b=config['learning_rate_competition_data'],
                              v=config['verbose'])

        predictions = net.predict_test_data()
        np.save(file=config[net_to_train]['predictions_path'], arr=predictions)
        print('saved model predictions at path: {}'.format(config[net_to_train]['predictions_path']))

def main_predict(config):
    predictions = []

    for net_type in config['net_types']:
        preds = np.load(config[net_type]['predictions_path'], allow_pickle=True)
        predictions.append(preds)
    mean_ensemble = np.mean(np.array(predictions), axis=0)
    np.save(file=config['final_predictions_path'], arr=mean_ensemble)
    print('saved final mean ensemble predictions at path: {}'.format(config['final_predictions_path']))

    dummy_model = NNet(data_paths=config['data_paths'])
    dummy_model.test_images_predictions = mean_ensemble
    mean_sub = dummy_model.create_submission_file(path=config['submission_path'], treshold=config['treshold'])

    dummy_model.display_test_predictions(config['submission_path'])
    dummy_model.display_test_predictions(config['submission_path'], samples_number=94, figure_size=(20, 470))
    plt.savefig(config['figures_path'])

def fix_config(config, path, net):
    if path != 'default':
        shutil.rmtree(config['submission_root'])

        config['submission_root'] = '../submissions/{}/'.format(path)
        config['submission_path'] = config['submission_root'] + 'submission.csv'
        config['figures_path'] = config['submission_root'] + 'predictions.png'
        config['checkpoint_root'] = config['submission_root'] + 'checkpoints/'
        config['prediction_root'] = config['submission_root'] + 'predictions/'
        config['final_predictions_path'] = config['prediction_root'] + '{}_predictions.npy'.format('final_ensemble')
        os.makedirs(config['checkpoint_root'], exist_ok=True)
        os.makedirs(config['prediction_root'], exist_ok=True)

        for net_type in config['net_types']:
            config[net_type]['checkpoint'] = config['checkpoint_root'] + '{}_weights.npy'.format(net_type)
            config[net_type]['predictions_path'] = config['prediction_root'] + '{}_predictions.npy'.format(net_type)
    
    if net != 'all':
        for net_type in config['net_types']:
            if net_type != net:
                del config[net_type]

        config['net_types'] = [net]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--save_path', dest='save_path', type=str, default='default', help='name of dir where to store training outputs')
    parser.add_argument('-n', '--net_to_train', dest='net_name', type=str, default='all', help='name of the network to train')
    parser.add_argument('-s', '--scope', dest='scope', type=str, default='train', help='scope of the job - can be either train or predict')
    args = parser.parse_args()

    save_path = str(args.save_path)
    net = str(args.net_name)
    scope = str(args.scope)

    assert net in ['all', 'u_xception', 'ures_xception', 'uspp_xception', 'u_resnet50v2', 'ures_resnet50v2', 'uspp_resnet50v2'], "net_to_train must be one of ['all', 'u_xception', 'ures_xception', 'uspp_xception', 'u_resnet50v2', 'ures_resnet50v2', 'uspp_resnet50v2']"
    assert scope in ['train', 'predict'], "scope must be one between train or predict"

    if save_path != 'default' or net != 'all':
        fix_config(config, save_path, net)
        if scope == 'train':
            with open(config['submission_root'] + 'config_{}.json'.format(net), 'w') as fp:
                json.dump(config, fp)

    if scope == 'train':
        main_train(config)
    elif scope == 'predict':
        main_predict(config)