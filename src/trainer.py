from config import config
from model import NNet
from utils import single_model_training
import numpy as np
import matplotlib.pyplot as plt

for net_to_train in config['net_types']:
    net = NNet(net_type=net_to_train, data_paths=config['data_paths'])
    single_model_training(model=net,
                          save_path=config[net_to_train]['checkpoint'],
                          additional_epochs=config['additional_epochs'],
                          competition_epochs=config['competition_epochs'],
                          b_size=config[net_to_train]['batch_size'],
                          loss=config['loss'],
                          l_rate_a=config['learning_rate_additional_data'],
                          l_rate_b=config['learning_rate_competition_data'])

    predictions = net.predict_test_data()
    np.save(file=config[net_to_train]['predictions_path'], arr=predictions)
    print('saved model predictions at path: {}'.format(config[net_to_train]['predictions_path']))

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