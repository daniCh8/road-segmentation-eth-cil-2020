import argparse
import shutil
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from config import config
from model import NNet
from utils import single_model_training, save_predictions_pdf


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
        sub = net.create_submission_file(path=config[net_to_train]['csv_path'], treshold=config['treshold'])
        print('saved model csv at path: {}'.format(config[net_to_train]['csv_path']))


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

    save_predictions_pdf(dummy_model, config)


def fix_config(config, path, nets):
    model_id = config['model_id']
    if path != 'default':
        shutil.rmtree(config['submission_root'])
        model_id = path

        config['submission_root'] = '../submissions/{}/'.format(path)
        config['submission_path'] = config['submission_root'] + 'submission.csv'
        config['figures_pdf'] = config['submission_root'] + 'predictions.pdf'
        config['checkpoint_root'] = config['submission_root'] + 'checkpoints/'
        config['prediction_root'] = config['submission_root'] + 'predictions/'
        config['csv_root'] = config['submission_root'] + 'csvs/'
        config['final_predictions_path'] = config['prediction_root'] + '{}_predictions.npy'.format('final_ensemble')
        os.makedirs(config['checkpoint_root'], exist_ok=True)
        os.makedirs(config['prediction_root'], exist_ok=True)
        os.makedirs(config['csv_root'], exist_ok=True)

        for net_type in config['net_types']:
            config[net_type]['checkpoint'] = config['checkpoint_root'] + '{}_weights.npy'.format(net_type)
            config[net_type]['predictions_path'] = config['prediction_root'] + '{}_predictions.npy'.format(net_type)
            config[net_type]['csv_path'] = config['csv_root'] + '{}_csv.csv'.format(net_type)
    
    if nets != []:
        if path == 'default':
            os.remove(config['submission_root'] + 'config.json')
        for net_type in nets:
            if net_type not in config['net_types']:
                config['net_types'].append(net_type)
                config[net_type] = {} 
                config[net_type]['batch_size'] = 2
                config[net_type]['checkpoint'] = config['checkpoint_root'] + '{}_weights.npy'.format(net_type)
                config[net_type]['predictions_path'] = config['prediction_root'] + '{}_predictions.npy'.format(net_type)
                config[net_type]['csv_path'] = config['csv_root'] + '{}_csv.csv'.format(net_type)
        for net_type in config['net_types']:
            if net_type not in nets:
                del config[net_type]

        config['net_types'] = nets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--save_path', dest='save_path', type=str, default='default', help='name of dir where to store training outputs')
    parser.add_argument('-n', '--nets_to_train', dest='net_names', action='append', default=[], help='list of names of the networks to train')
    parser.add_argument('-s', '--scope', dest='scope', type=str, default='train', help='scope of the job - can be either train or predict')
    parser.add_argument('-a', '--add_epochs', dest='additional_epochs', type=int, default=-1, help='sets the number of epochs used to train the nets on the google data')
    parser.add_argument('-c', '--comp_epochs', dest='competition_epochs', type=int, default=-1, help='sets the number of epochs used to train the nets on the competition data')
    args = parser.parse_args()

    save_path = str(args.save_path)
    nets = list(args.net_names)
    scope = str(args.scope)
    additional_epochs = int(args.additional_epochs)
    competition_epochs = int(args.competition_epochs)

    assert nets==['all'] or set(nets).issubset(set(['u_xception', 'ures_xception', 'uspp_xception', 'u_resnet50v2', 'ures_resnet50v2', 'uspp_resnet50v2', 'deepuresnet', 'dunet'])), "nets_to_train must be a subset of ['u_xception', 'ures_xception', 'uspp_xception', 'u_resnet50v2', 'ures_resnet50v2', 'uspp_resnet50v2', 'deepuresnet', 'dunet']"
    assert scope in ['train', 'predict'], "scope must be one between train or predict"
    
    if additional_epochs != -1 or competition_epochs != -1:
        if additional_epochs != -1:
            assert additional_epochs > 0, "epochs number can't be lower or equal than 0"
            config['additional_epochs'] = additional_epochs

        if competition_epochs != -1:
            assert competition_epochs > 0, "epochs number can't be lower or equal than 0"
            config['competition_epochs'] = competition_epochs
        
        os.remove(config['submission_root'] + 'config.json')
        with open(config['submission_root'] + 'config.json', 'w') as fp:
            json.dump(config, fp)
    
    if save_path != 'default' or nets != []:
        id = config['model_id']
        fix_config(config, save_path, nets)
        if scope == 'train':
            with open(config['submission_root'] + 'config_{}.json'.format(id), 'w') as fp:
                json.dump(config, fp)

    if scope == 'train':
        main_train(config)
        if nets == []:
            main_predict(config)
    elif scope == 'predict':
        main_predict(config)
    