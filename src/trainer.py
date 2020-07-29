import argparse
import shutil
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from config import create_config
from model import NNet
from utils import single_model_training, save_predictions_pdf


def main_train(config):
    for net_to_train, net_name in zip(config['net_types'], config['net_names']):
        net = NNet(net_type=net_to_train, data_paths=config['data_paths'])
        single_model_training(model=net,
                              save_path=config[net_name]['checkpoint'],
                              additional_epochs=config[net_name]['additional_epochs'],
                              competition_epochs=config[net_name]['competition_epochs'],
                              b_size=config[net_name]['batch_size'],
                              loss=config['loss'],
                              l_rate_a=config['learning_rate_additional_data'],
                              l_rate_b=config['learning_rate_competition_data'],
                              v=config['verbose'])

        predictions = net.predict_test_data()
        np.save(file=config[net_name]['predictions_path'], arr=predictions)
        print('saved model predictions at path: {}'.format(config[net_name]['predictions_path']))
        sub = net.create_submission_file(path=config[net_name]['csv_path'], treshold=config['treshold'])
        print('saved model csv at path: {}'.format(config[net_name]['csv_path']))


def main_predict(config):
    predictions = []

    for net_name in config['net_names']:
        preds = np.load(config[net_name]['predictions_path'], allow_pickle=True)
        predictions.append(preds)
    mean_ensemble = np.mean(np.array(predictions), axis=0)
    np.save(file=config['final_predictions_path'], arr=mean_ensemble)
    print('saved final mean ensemble predictions at path: {}'.format(config['final_predictions_path']))

    dummy_model = NNet(data_paths=config['data_paths'])
    dummy_model.test_images_predictions = mean_ensemble
    mean_sub = dummy_model.create_submission_file(path=config['submission_path'], treshold=config['treshold'])

    save_predictions_pdf(dummy_model, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--save_path', dest='save_path', type=str, default='', help='name of dir where to store training outputs')
    parser.add_argument('-n', '--nets_to_train', dest='net_names', action='append', default=[], help='list of names of the networks to train')
    parser.add_argument('-s', '--scope', dest='scope', type=str, default='train+predict', help='scope of the job - can be either train or predict')
    parser.add_argument('-a', '--add_epochs', dest='additional_epochs', type=int, action='append', default=[], help='list of number of epochs used to train the nets on the google data')
    parser.add_argument('-c', '--comp_epochs', dest='competition_epochs', type=int, action='append', default=[], help='list of number of epochs used to train the nets on the competition data')
    parser.add_argument('-b', '--batch_sizes', dest='batch_sizes', type=int, action='append', default=[], help='list of batch sizes used to train the nets')
    parser.add_argument('-t', '--treshold', dest='treshold', type=float, default=.4, help='treshold to use to make submissions')
    args = parser.parse_args()

    save_path = str(args.save_path)
    nets = list(args.net_names)
    scope = str(args.scope)
    additional_epochs = list(args.additional_epochs)
    competition_epochs = list(args.competition_epochs)
    batch_sizes = list(args.batch_sizes)
    treshold = float(args.treshold)

    assert len(nets)==0 or set(nets).issubset(set(['u_xception', 'u_resnet50v2', 'unet'])), "nets_to_train must be a subset of ['u_xception', 'u_resnet50v2', 'unet']"
    assert scope in ['train', 'predict', 'train+predict'], "scope must be one between train, predict, train+predict"
    assert treshold>=0.0 and treshold<=1.0, "treshold must be a float between 0. and 1.0"

    for epochs in additional_epochs:
        assert epochs > 0, "epochs numbers can't be lower or equal than 0"
    for epochs in competition_epochs:
        assert epochs > 0, "epochs numbers can't be lower or equal than 0"
    for batch_size in batch_sizes:
        assert epochs > 0, "batch sizes can't be lower or equal than 0"
    
    config = create_config(nets, save_path, batch_sizes, additional_epochs, competition_epochs, treshold)

    if scope == 'train' or 'train+predict':
        main_train(config)
        if scope == 'train+predict':
            main_predict(config)
    elif scope == 'predict':
        main_predict(config)
    