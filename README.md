# [Computational Intelligence Lab](http://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lang=en&lerneinheitId=135225&semkez=2020S&ansicht=KATALOGDATEN&) 2020, Course Project
## Team 
- ### Daniele Chiappalupi ([@daniCh8](https://github.com/daniCh8))<br>dchiappal@student.ethz.ch
- ### Elena Iannucci ([@eleiannu](https://github.com/eleiannu))<br>eiannucci@student.ethz.ch
- ### Samuele Piazzetta ([@piaz97](https://github.com/piaz97))<br>samuele.piazzetta@gmail.com
- ### Gianluca Lain ([@OsD977](https://github.com/OsD977))<br>gianluca.lain97@gmail.com

## Project Description
The goal of the project is to create a model able to detect and extract road networks from aerial images. We tried many different networks, and then built an ensemble of the most accurate ones. Here is a sample prediction of our ensemble:

![sample_prediction](https://i.postimg.cc/dt55wPQS/cropped-sample-prediction.png)

## Network Types
### [U-Xception Net](/src/nets/uxception.py)
It's a U-Net that uses an Xception Net pretrained on the 'imagenet' dataset as encoder. The intermediate outputs of the Xception Net are taken as they are and fed to the decoders.

### [U-Res-Xception Net](/src/nets/uresxception.py)
It's another U-Net that uses an Xception Net pretrained on the 'imagenet' dataset as encoder, but this time the intermediate outputs of the Xception Net are processed by two residual blocks before being fed to the decoders.

### [U-Xception Net with Spatial Pyramid Pooling](/src/nets/uresxceptionsp.py)
It's the same architecture of the U-Res-Xception Net, but instead of being processed by two residual blocks, the Xception outputs are refined by Spatial Pyramid blocks.

### [U-ResNet](/src/nets/uresnet50v2.py)
It's a U-Net that uses a pretrained ResNet50V2 as encoder. Like in the U-Res-Xception Net, the intermediate outputs of the pretrained net are processed by two residual blocks before being fed to the decoders.

### [DeepRes-UNet](/src/nets/deepresunet.py)
It's a deep U-Net that does not use any pretrained net as encoder, but only residual blocks.

### [D-UNet](/src/nets/dunet.py)
It's a dimension fusion U-Net, that processed the input both in 4D and 3D, before mixing all together.

## Usage
All the modules are python files, whereas the main files are Jupyter Notebooks. Any single network can be trained and evaluated through [single_model_trainer.ipynb](/src/single_model_trainer.ipynb). Note that Jupyter Notebooks are useful to visualize data, but the training and checkpointing process is actually all handled by [model.py](/src/model.py), that contains the class `NNet`:

	__init__(self, val_split=.0, model_to_load='None', net_type='uxception')

`model_to_load` can be used to load any pretrained model. If it's `'None'`, a new network of type `net_type` will be created.

After training all the models of interest, an ensemble of them can be created through [ensemble_predictions.ipynb](/src/ensemble_predictions.ipynb).

## Additional Data
We gathered roughly 1000 additional images using the Google Maps API. The Jupyter Notebook we used to do so is available [here](/additional_maps_data.ipynb). Note that you will need an API key to be able to run it.

We also used the [albumentations package](https://github.com/albumentations-team/albumentations) to augment the training dataset and make the networks able to generalize more. Those data augmentations are handled by the [DataGenerator](/src/DataGenerator.py) class.