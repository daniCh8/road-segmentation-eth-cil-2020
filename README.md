# [Computational Intelligence Lab](http://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lang=en&lerneinheitId=135225&semkez=2020S&ansicht=KATALOGDATEN&) 2020, Course Project
## Team 
- ### Daniele Chiappalupi ([@daniCh8](https://github.com/daniCh8))<br>dchiappal@student.ethz.ch
- ### Elena Iannucci ([@eleiannu](https://github.com/eleiannu))<br>eiannucci@student.ethz.ch
- ### Samuele Piazzetta ([@piaz97](https://github.com/piaz97))<br>samuele.piazzetta@gmail.com
- ### Gianluca Lain ([@OsD977](https://github.com/OsD977))<br>gianluca.lain97@gmail.com

## Project Description
The goal of the project is to create a model able to detect and extract road networks from aerial images. We tried many different networks, and then built an ensemble of the most accurate ones. Here there is a sample prediction of our ensemble:

![sample_prediction](https://i.postimg.cc/dt55wPQS/cropped-sample-prediction.png)

The final outputs of our network on test data can be found [here](/predictions.pdf).

## Network Types
All the networks used for the final ensemble (the first six below) are based on the same architecture. Here is a scheme of such architecture:

![net_architecture](https://i.ibb.co/bKZK4nH/net-ushape-w-legend-18.png)

As shown by the graph, the skeleton is basically a U-Net that uses a pre-trained model on the encoder part. What vary between our different networks are such pre-trained models and the decoder/encoder blocks used to process the intermediate outputs.

We used two types of blocks: [Spatial Pyramid Pooling blocks](https://arxiv.org/abs/1406.4729) and [Residual blocks](https://arxiv.org/abs/1512.03385). Below is a sketch of them.

![blocks](https://i.ibb.co/TLY2xzw/blocks-legend-v4.png)

So, now that we described the general architecture of our networks, below are listed the different types and implementations of them.
 
### [U-Xception Net](/src/nets/u_xception.py)
It's a U-Net that uses an Xception Net pretrained on the 'imagenet' dataset as encoder. The intermediate outputs of the Xception Net are taken as they are and fed to the decoders.

### [URES-Xception](/src/nets/ures_xception.py)
It's another U-Net that uses an Xception Net pretrained on the 'imagenet' dataset as encoder, but this time the intermediate outputs of the Xception Net are processed by two residual blocks before being fed to the decoders.

### [USPP-Xception](/src/nets/uspp_xception.py)
It's the same architecture of the network above, but instead of being processed by two residual blocks, the Xception outputs are refined by Spatial Pyramid blocks.

### [U-ResNet50V2 Net](/src/nets/u_resnet50v2.py)
It's a U-Net that uses a ResNet50V2 pretrained on the 'imagenet' dataset as encoder. The intermediate outputs of the ResNet50V2 are vanilla fed to the decoders.

### [URES-ResNet50V2 Net](/src/nets/ures_resnet50v2.py)
It's a U-Net that uses a pretrained ResNet50V2 as encoder. Like in the URES-Xception Net, the intermediate outputs of the pretrained net are processed by two residual blocks before being fed to the decoders.

### [USPP-ResNet50V2 Net](/src/nets/uspp_resnet50v2.py)
It's the same architecture of the network above, but instead of being processed by two residual blocks, the ResNetV2 outputs are refined by Spatial Pyramid blocks.

### [DeepRes-UNet](/src/nets/deepresunet.py)
It's a deep U-Net that does not use any pretrained net as encoder, but only residual blocks.

### [D-UNet](/src/nets/dunet.py)
It's a dimension fusion U-Net, that process the input both in 4D and 3D, before mixing all together.

## Usage
All the modules are python files, whereas the main files are Jupyter Notebooks. Any single network can be trained and evaluated through [single_model_trainer.ipynb](/src/single_model_trainer.ipynb). Note that Jupyter Notebooks are useful to visualize data, but the training and checkpointing process is actually all handled by [model.py](/src/model.py), that contains the class `NNet`:

```python
	__init__(self, val_split=.0, model_to_load='None', net_type='u_xception', load_weights='None')
```

`model_to_load` can be used to load any pretrained model. If it's `'None'`, a new network of type `net_type` will be created. In such case, `load_weights` can be used to recover the weights of a trained model. Obviously, if `load_weights` is not `'None'`, the weights must be coherent with the `net_type` created.

After training all the models of interest, an ensemble of them can be created through [ensemble_predictions.ipynb](/src/ensemble_predictions.ipynb).

## Additional Data
We gathered roughly 1000 additional images using the Google Maps API. The Jupyter Notebook we used to do so is available [here](/additional_maps_data.ipynb). Note that you will need an API key to be able to run it.

We also used the [albumentations package](https://github.com/albumentations-team/albumentations) to augment the training dataset and make the networks able to generalize more. Those data augmentations are handled by the [DataGenerator](/src/DataGenerator.py) class.

## Final Prediction
In order to obtain the predictions, we used a mean ensemble of six models. Those models were the following:

- [U-Xception Net](/src/nets/u_xception.py)
- [URES-Xception Net](/src/nets/ures_xception.py)
- [USPP-Xception Net](/src/nets/uspp_xception.py)
- [U-ResNet50V2](/src/nets/u_resnet50v2.py)
- [URES-ResNet50V2](/src/nets/ures_resnet50v2.py)
- [USPP-ResNet50V2](/src/nets/uspp_resnet50v2.py)

Each model has been first trained for 20 epochs on the additional data with `learning rate = .0001`, and then fine-tuned for 60 epochs on the competition data with `learning rate = .00001`. During both training and fine-tuning all of the aforementioned data augmentation have been used. Moreover, during the `fit` of the networks, both the callbacks *Early Stopping* and *Learning Rate reduction on Plateau* were on. The whole pipeline of fitting can be found [here](/src/single_model_trainer.ipynb).

After training all the models, a mean ensemble of their predictions has been created [here](/src/ensemble_predictions.ipynb). Note that the averaging is made on the outputs of the neural networks, and not on the final binary submission values.

Finally, the submission `csv` file is created with the parameter  `treshold = .4`.

Note that the test images have a shape of  `608*608*3`, whereas the training images are  `400*400*3`. In order to make the predictions, we cut the test images in four squares of  size `400*400*3`, and then recomposed the full prediction merging those blocks, averaging the pixels in common. This is done in the function  `merge prediction`, which can be found in the [utils](/src/utils.py) module.

## Train and Predict in a single run
We uploaded a single notebook that can be run in order to train the whole ensemble from scratch [here](/src/all_in_one_predictor.ipynb). Note that you will still need the Google Maps API data, that we can't upload for copyright reasons. Running all the cells generates a new submission. All the parameters can be tuned in this [config](/src/config.py) file:

- `loss` controls which loss function will be used to train the single networks. Available loss functions are [dice loss](https://arxiv.org/abs/1911.02855) and [binary_cross_entropy](https://en.wikipedia.org/wiki/Cross_entropy).
- `net_types` sets which nets will be used in the network. It must be a subset of: `['u_xception', 'ures_xception', 'uspp_xception', 'deepuresnet', 'u_resnet50v2',  'ures_resnet50v2', 'uspp_resnet50v2', 'dunet']`.
- `additional_epochs` sets the number of epochs that the network will train on the additional Google Maps API data.
- `competition_epochs` sets the number of epochs that the network will train on the competition data.
- `learning_rate_additional_data` sets the learning rate that will be used during the training on the additional Google Maps API data.
- `learning_rate_competition_data` sets the learning rate that will be used during the training on the competition data.
- `treshold` sets the treshold that will be used to compute the final submission on 16*16 batches.
- `batch_size` controls the batch_size that will be used during the training. There's a different one for each net, in order to be able to reduce the batch size of the bigger networks and not run into a `ResourceExhausted` failure.
- `model_id` saves the id of the network, so that every run will be saved in a different path.
- `submission_path` controls where the submission file will be stored.
- `checkpoint_root` controls where the single networks weight checkpoint files will be stored.
- `predictions_path` controls where the single model predictions to be averaged will be stored.

A json dump of the configurations for every run will also be stored in the submission directory, so that every run is bind with its parameters. Below is an example of `config` file, in json syntax (dumped from a project run).

```javascript
{
  "net_types": ["u_xception", "ures_xception", "uspp_xception", "u_resnet50v2", "ures_resnet50v2", "uspp_resnet50v2"],
  "additional_epochs": 30,
  "competition_epochs": 60,
  "loss": "dice",
  "learning_rate_additional_data": 0.0001,
  "learning_rate_competition_data": 1e-05,
  "treshold": 0.4,
  "model_id": "submission_30-05-2020,08-11",
  "submission_root": "../submissions/submission_30-05-2020,08-11/",
  "submission_path": "../submissions/submission_30-05-2020,08-11/submission.csv",
  "checkpoint_root": "../submissions/submission_30-05-2020,08-11/checkpoints/",
  "prediction_root": "../submissions/submission_30-05-2020,08-11/predictions/"
  "u_xception": 
  {
	"batch_size": 8,
	"checkpoint": "../submissions/submission_30-05-2020,08-11/checkpoints/u_xception_weights.npy",
	"predictions_path": "../submissions/submission_30-05-2020,08-11/predictions/u_xception_predictions.npy"
  },
  "ures_xception":
  {
	"batch_size": 6,
	"checkpoint": "../submissions/submission_30-05-2020,08-11/checkpoints/ures_xception_weights.npy",
	"predictions_path": "../submissions/submission_30-05-2020,08-11/predictions/ures_xception_predictions.npy"
  },
  "uspp_xception":
  {
	"batch_size": 6,
	"checkpoint": "../submissions/submission_30-05-2020,08-11/checkpoints/uspp_xception_weights.npy",
	"predictions_path": "../submissions/submission_30-05-2020,08-11/predictions/uspp_xception_predictions.npy"
  },
  "u_resnet50v2":
  {
	"batch_size": 8,
	"checkpoint": "../submissions/submission_30-05-2020,08-11/checkpoints/u_resnet50v2_weights.npy",
	"predictions_path": "../submissions/submission_30-05-2020,08-11/predictions/u_resnet50v2_predictions.npy"
  },
  "ures_resnet50v2":
  {
	"batch_size": 6,
	"checkpoint": "../submissions/submission_30-05-2020,08-11/checkpoints/ures_resnet50v2_weights.npy",
	"predictions_path": "../submissions/submission_30-05-2020,08-11/predictions/ures_resnet50v2_predictions.npy"
  },
  "uspp_resnet50v2":
  {
	"batch_size": 4,
	"checkpoint": "../submissions/submission_30-05-2020,08-11/checkpoints/uspp_resnet50v2_weights.npy",
	"predictions_path": "../submissions/submission_30-05-2020,08-11/predictions/uspp_resnet50v2_predictions.npy"
  }
}
```

Note that the models checkpoints are stored as numpy files. Indeed, we're not actually storing `keras`' checkpoints, but the weights of the networks. This is done to speed-up both the saving and the loading model phases. A model saved this way can be loaded by creating a new `NNet` class, with parameters `net_type` equals to the type of network we want to load and `load_weights` equals to the path where the weights file of such model are stored.