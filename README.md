# [Computational Intelligence Lab](http://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lang=en&lerneinheitId=135225&semkez=2020S&ansicht=KATALOGDATEN&) 2020, Course Project

- [Computational Intelligence Lab 2020, Course Project](#computational-intelligence-lab-2020-course-project)
  - [Team](#team)
  - [Project Description](#project-description)
  - [Network Types](#network-types)
    - [U-Net](#u-net)
    - [U-Xception Net](#u-xception-net)
    - [URES-Xception](#ures-xception)
    - [USPP-Xception](#uspp-xception)
    - [U-ResNet50V2 Net](#u-resnet50v2-net)
    - [URES-ResNet50V2 Net](#ures-resnet50v2-net)
    - [USPP-ResNet50V2 Net](#uspp-resnet50v2-net)
    - [DeepRes-UNet](#deepres-unet)
    - [D-UNet](#d-unet)
  - [Additional Data](#additional-data)
  - [Final Predictions](#final-predictions)
  - [Usage](#usage)
  - [Requirements](#requirements)
  - [Usage in Leonhard](#usage-in-leonhard)

## Team 
-  **Daniele Chiappalupi** ([@daniCh8](https://github.com/daniCh8))<br>dchiappal@student.ethz.ch
-  **Elena Iannucci** ([@eleiannu](https://github.com/eleiannu))<br>eiannucci@student.ethz.ch
- **Samuele Piazzetta** ([@piaz97](https://github.com/piaz97))<br>samuele.piazzetta@gmail.com
- **Gianluca Lain** ([@OsD977](https://github.com/OsD977))<br>gianluca.lain97@gmail.com

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

### [U-Net](/src/nets/unet.py)
It's a plain U-Net. No pretrained networks is used as encoder and no famous block is used to process the intermediate outputs.

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

## Additional Data
We gathered roughly 1000 additional images using the Google Maps API. The Jupyter Notebook we used to do so is available [here](/additional_maps_data.ipynb). Note that you will need an API key to be able to run it.

We also used the [albumentations package](https://github.com/albumentations-team/albumentations) to augment the training dataset and make the networks able to generalize more. Those data augmentations are handled by the [DataGenerator](/src/DataGenerator.py) class.

## Final Predictions
In order to obtain the predictions, we used a mean ensemble of six models. Those models were the following:

- [U-Xception Net](/src/nets/u_xception.py)
- [URES-Xception Net](/src/nets/ures_xception.py)
- [USPP-Xception Net](/src/nets/uspp_xception.py)
- [U-ResNet50V2](/src/nets/u_resnet50v2.py)
- [URES-ResNet50V2](/src/nets/ures_resnet50v2.py)
- [USPP-ResNet50V2](/src/nets/uspp_resnet50v2.py)

Each model has been first trained for 20 epochs on the additional data with `learning rate = .0001`, and then fine-tuned for 60 epochs on the competition data with `learning rate = .00001`. During both training and fine-tuning all of the aforementioned data augmentation have been used. Moreover, during the `fit` of the networks, both the callbacks *Early Stopping* and *Learning Rate reduction on Plateau* were on. The whole pipeline of fitting can be found [here](/src/utils.py), in the method `single_model_training` (line 61).

After training all the models, a mean ensemble of their predictions is created. Note that the averaging is made on the outputs of the neural networks, and not on the final binary submission values.

Finally, the submission `csv` file is created with the parameter  `treshold = .4`.

Note that the test images have a shape of  `608*608*3`, whereas the training images are  `400*400*3`. In order to make the predictions, we cut the test images in four squares of  size `400*400*3`, and then recomposed the full prediction merging those blocks, averaging the pixels in common. This is done in the function  `merge prediction`, which can be found in the [utils](/src/utils.py) module.

## Usage
The whole ensemble can be trained from scratch either running all the cells in [this jupyter notebook](/src/all_in_one_predictor.ipynb) or using [this python file](/src/trainer.py) as explained below in the [Usage in Leonhard](#usage-in-leonhard) section. Note that you will still need the Google Maps API data, that we can't upload for copyright reasons.

The training and checkpointing process of the nets is all handled by [model.py](/src/model.py), that contains the class `NNet`:

```python
  __init__(self, val_split=.0, model_to_load='None', net_type='u_xception', load_weights='None', data_paths=None)
```

`model_to_load` can be used to load any pretrained model. If it's `'None'`, a new network of type `net_type` will be created. In such case, `load_weights` can be used to recover the weights of a trained model. Obviously, if `load_weights` is not `'None'`, the weights must be coherent with the `net_type` created. `data_paths` can be used to define custom data paths, otherwise the default relative ones will be used.

When using [all_in_one_predictor.ipynb](/src/all_in_one_predictor.ipynb) or [trainer.py](/src/trainer.py), all the project parameters can be tuned via this [config](/src/config.py) file. Here is an explanation of each one of them:

- `loss` controls which loss function will be used to train the single networks. Available loss functions are [dice loss](https://arxiv.org/abs/1911.02855) and [binary_cross_entropy](https://en.wikipedia.org/wiki/Cross_entropy).
- `net_types` sets which nets will be used in the network. It must be a subset of: `['u_xception', 'ures_xception', 'uspp_xception', 'deepuresnet', 'u_resnet50v2',  'ures_resnet50v2', 'uspp_resnet50v2', 'dunet', 'unet']`.
- `additional_epochs` sets the number of epochs that each network will train on the additional Google Maps API data.
- `competition_epochs` sets the number of epochs that each network will train on the competition data.
- `learning_rate_additional_data` sets the learning rate that will be used during the training on the additional Google Maps API data.
- `learning_rate_competition_data` sets the learning rate that will be used during the training on the competition data.
- `treshold` sets the treshold that will be used to compute the final submission on 16*16 batches.
- `data_paths` controls the paths where to find the training data.
- `model_id` saves the id of the network, so that every run will be saved in a different path.
- `submission_path` controls where the submission file will be stored.
- `csv_path` controls where the submission file of each single net will be stored.
- `checkpoint_root` controls where the single networks weight checkpoint files will be stored.
- `predictions_path` controls where the single model predictions to be averaged will be stored.
- `final_predictions_path` controls where the final ensemble model predictions.
- `batch_size` controls the batch_size that will be used during the training. There's a different one for each net, in order to be able to reduce the batch size of the bigger networks and not run into a `ResourceExhausted` failure.

A json dump of the configurations for every run will also be stored in the submission directory, so that every run is bind with its parameters. Below is an example of `config` file, in json syntax (dumped from a project run).

```rust
{
   "net_types":[
      "u_xception",
      "ures_xception",
      "uspp_xception",
      "u_resnet50v2",
      "ures_resnet50v2",
      "uspp_resnet50v2"
   ],
   "loss":"dice",
   "learning_rate_additional_data":0.0001,
   "learning_rate_competition_data":1e-05,
   "treshold":0.4,
   "verbose":2,
   "data_paths":{
      "data_dir":"../data/",
      "image_path":"../data/training/images/",
      "groundtruth_path":"../data/training/groundtruth/",
      "additional_images_path":"../data/additional_data/images/",
      "additional_masks_path":"../data/additional_data/masks/",
      "test_data_path":"../data/test_images/"
   },
   "model_id":"submission_19-07-2020,11-13",
   "submission_root":"../submissions/submission_19-07-2020,11-13/",
   "submission_path":"../submissions/submission_19-07-2020,11-13/submission.csv",
   "figures_pdf":"../submissions/submission_19-07-2020,11-13/predictions.pdf",
   "checkpoint_root":"../submissions/submission_19-07-2020,11-13/checkpoints/",
   "prediction_root":"../submissions/submission_19-07-2020,11-13/predictions/",
   "csv_root":"../submissions/submission_19-07-2020,11-13/csvs/",
   "final_predictions_path":"../submissions/submission_19-07-2020,11-13/predictions/final_ensemble_predictions.npy",
   "u_xception":{
      "batch_size":2,
      "additional_epochs":40,
      "competition_epochs":60,
      "checkpoint":"../submissions/submission_19-07-2020,11-13/checkpoints/0_u_xception_40g_60c_weights.npy",
      "predictions_path":"../submissions/submission_19-07-2020,11-13/predictions/0_u_xception_40g_60c_predictions.npy",
      "csv_path":"../submissions/submission_19-07-2020,11-13/csvs/0_u_xception_40g_60c_csv.csv"
   },
   "ures_xception":{
      "batch_size":2,
      "additional_epochs":40,
      "competition_epochs":60,
      "checkpoint":"../submissions/submission_19-07-2020,11-13/checkpoints/1_ures_xception_40g_60c_weights.npy",
      "predictions_path":"../submissions/submission_19-07-2020,11-13/predictions/1_ures_xception_40g_60c_predictions.npy",
      "csv_path":"../submissions/submission_19-07-2020,11-13/csvs/1_ures_xception_40g_60c_csv.csv"
   },
   "uspp_xception":{
      "batch_size":2,
      "additional_epochs":40,
      "competition_epochs":60,
      "checkpoint":"../submissions/submission_19-07-2020,11-13/checkpoints/2_uspp_xception_40g_60c_weights.npy",
      "predictions_path":"../submissions/submission_19-07-2020,11-13/predictions/2_uspp_xception_40g_60c_predictions.npy",
      "csv_path":"../submissions/submission_19-07-2020,11-13/csvs/2_uspp_xception_40g_60c_csv.csv"
   },
   "u_resnet50v2":{
      "batch_size":2,
      "additional_epochs":40,
      "competition_epochs":60,
      "checkpoint":"../submissions/submission_19-07-2020,11-13/checkpoints/3_u_resnet50v2_40g_60c_weights.npy",
      "predictions_path":"../submissions/submission_19-07-2020,11-13/predictions/3_u_resnet50v2_40g_60c_predictions.npy",
      "csv_path":"../submissions/submission_19-07-2020,11-13/csvs/3_u_resnet50v2_40g_60c_csv.csv"
   },
   "ures_resnet50v2":{
      "batch_size":2,
      "additional_epochs":40,
      "competition_epochs":60,
      "checkpoint":"../submissions/submission_19-07-2020,11-13/checkpoints/4_ures_resnet50v2_40g_60c_weights.npy",
      "predictions_path":"../submissions/submission_19-07-2020,11-13/predictions/4_ures_resnet50v2_40g_60c_predictions.npy",
      "csv_path":"../submissions/submission_19-07-2020,11-13/csvs/4_ures_resnet50v2_40g_60c_csv.csv"
   },
   "uspp_resnet50v2":{
      "batch_size":2,
      "additional_epochs":40,
      "competition_epochs":60,
      "checkpoint":"../submissions/submission_19-07-2020,11-13/checkpoints/5_uspp_resnet50v2_40g_60c_weights.npy",
      "predictions_path":"../submissions/submission_19-07-2020,11-13/predictions/5_uspp_resnet50v2_40g_60c_predictions.npy",
      "csv_path":"../submissions/submission_19-07-2020,11-13/csvs/5_uspp_resnet50v2_40g_60c_csv.csv"
   }
}
```

Note that the models checkpoints are stored as numpy files. Indeed, we're not actually storing `keras`' checkpoints, but the weights of the networks. This is done to speed-up both the saving and the loading model phases. A model saved this way can be loaded by creating a new `NNet` class, with parameters `net_type` equals to the type of network we want to load and `load_weights` equals to the path where the weights file of such model are stored.

## Requirements

The external libraries we used are listed in the [setup](/src/setup.py) file. Here is a recap:
- `tensorflow-gpu==1.15.2`
- `keras==2.3.1`
- `matplotlib`
- `numpy`
- `pandas`
- `scikit-learn`
- `scikit-image`
- `albumentations`
- `tqdm`

All of those packages can be easily installed using `pip`.

## Usage in [Leonhard](https://scicomp.ethz.ch/wiki/Leonhard)

Here is the workflow we followed to train the ensemble inside the Leonhard cluster.

We need to load the required modules including python, CUDA and cuDNN provided on the server:

```bash
module load python_gpu/3.6.4 hdf5 eth_proxy
module load cudnn/7.2
```

Then we'll create a virtual environment in order to install and save the required dependencies. We'll use `virtualenvwrapper` to manage our virtual environments on the cluster:

```bash
pip install virtualenvwrapper
export VIRTUALENVWRAPPER_PYTHON=/cluster/apps/python/3.6.4/bin/python
export WORKON_HOME=$HOME/.virtualenvs
source $HOME/.local/bin/virtualenvwrapper.sh
```

After completing the installation, we'll create the environment. Here we'll call it `pochi-ma-pochi` like our team name:

```bash
mkvirtualenv "pochi-ma-pochi"
```

The newly created virtual environment will be activated by default.
It might be a good idea to add this set of commands to the `.bashrc` file in order to not have to run them every time we access to the cluster. To do so, just add the following lines at the end of `~/.bashrc`:

```bash
module load python_gpu/3.6.4 hdf5 eth_proxy
module load cudnn/7.6.4
module load cuda/10.0.130
export VIRTUALENVWRAPPER_PYTHON=/cluster/apps/python/3.6.4/bin/python
export WORKON_HOME=$HOME/.virtualenvs
source $HOME/.local/bin/virtualenvwrapper.sh
workon "pochi-ma-pochi"
```

We'll now need to install all the required dependencies, and make sure to not have any version incompatibilities between different packages. Cd to the [src](/src) folder and run:

```bash
python setup.py install
pip install --upgrade scikit-image
pip install --upgrade numpy
pip install --upgrade scipy
pip install --upgrade scikit-learn
```

We are now all set to run our networks.
Let's check that everything went fine: we'll run an interactive GPU environment to see if the tensorflow and keras versions are the right ones:

```bash
bsub -Is -n 1 -W 1:00 -R "rusage[mem=4096, ngpus_excl_p=1]" bash
```

We'll have to wait some time for the dispatch. Once we are inside, we'll run the following commands to check the versions of tensorflow and keras. Also, it's important to check that we are actually using the GPU:

```bash
python
import keras
keras.__version__
import tensorflow as tf
tf.__version__
tf.config.experimental.list_physical_devices('GPU')
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
exit()
exit
```

If everything was correctly set, the full output of this interactive session should be the following:

```rust
> (pochi-ma-pochi) [dchiappal@lo-login-01 ~]$ bsub -Is -n 1 -W 1:00 -R "rusage[mem=4096, ngpus_excl_p=1]" bash
Generic job.
Job <7033720> is submitted to queue <gpu.4h>.
<<Waiting for dispatch ...>>
<<Starting on lo-s4-020>>

The following have been reloaded with a version change:
  1) cuda/9.0.176 => cuda/10.0.130     2) cudnn/7.0 => cudnn/7.6.4

> (pochi-ma-pochi) [dchiappal@lo-s4-020 ~]$ python
Python 3.6.4 (default, Apr 10 2018, 08:00:27)
[GCC 4.8.5 20150623 (Red Hat 4.8.5-16)] on linux
Type "help", "copyright", "credits" or "license" for more information.
> >>> import keras
Using TensorFlow backend.
> >>> keras.__version__
'2.3.1'
> >>> import tensorflow as tf
> >>> tf.__version__
'1.15.2'
> >>> tf.config.experimental.list_physical_devices('GPU')
2020-07-11 19:38:13.519467: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-07-11 19:38:13.598754: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1639] Found device 0 with properties:
name: GeForce GTX 1080 major: 6 minor: 1 memoryClockRate(GHz): 1.7335
pciBusID: 0000:0e:00.0
2020-07-11 19:38:13.599744: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-07-11 19:38:13.602673: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2020-07-11 19:38:13.605091: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2020-07-11 19:38:13.605878: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2020-07-11 19:38:13.608723: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2020-07-11 19:38:13.610998: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2020-07-11 19:38:13.615122: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-07-11 19:38:13.621832: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1767] Adding visible gpu devices: 0
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
> >>> sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
[...]
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX 1080, pci bus id: 0000:0e:00.0, compute capability: 6.1

> >>> exit()
> (pochi-ma-pochi) [dchiappal@lo-s4-020 ~]$ exit
```

Finally, we can submit our project to the GPU queue (always remembering to set valid data paths in [config](/src/config.py)). This is the submission command we used:

```sh
bsub -n 8 -W 12:00 -o project_log -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" python trainer.py [-a --add_epochs] [-c --comp_epochs] [-n --nets_to_train] [-p --save_path] [-s --scope]
```

Let's break down the arguments of the call:
- `-n 8` means that we are requesting 8 CPUs;
- `-W 12:00` means that the job can't last more than 12 hours. This makes it go into the 24h queue of the cluster.
- `-o logs/x` means that the output of the job will be stored into the file `./logs/x`.
- `-R "rusage[mem=8192, ngpus_excl_p=1]"` describes how much memory we request per CPU (8GB) and how many GPUs we ask (1).
- `-R "select[gpu_model0==GeForceRTX2080Ti]"` explicitly requests a RTX2080Ti GPU for the job. We use it to speed up the run.
- `-n` can be used to set a different subset of nets to train rather than the default one. The nets must be passed iteratively (i.e. `-n u_xception -n ures_xception -n uspp_xception` will train the subset of nets: `['u_xception', 'ures_xception', 'uspp_xception']`). The nets passed must be a subset of `['u_xception', 'ures_xception', 'uspp_xception', 'u_resnet50v2', 'ures_resnet50v2', 'uspp_resnet50v2', 'deepuresnet', 'dunet', 'unet']`.
- `-a` can be used to set the number of additional epochs used to train the networks. The number of values given must be the same as the number of nets to train, and the order counts. An example would be: `-n u_xception -n u_xception -a 20 -a 30`: such commands will train two u_xceptions, one with 20 epochs on Google Data and the other one with 30 epochs on Google Data. If no value is specified, the default values in [config](/src/config.py) will be used.
- `-c` can be used to set the number of competition epochs used to train the networks. The number of values given must be the same as the number of nets to train, and the order counts: the mechanism is the same as the one above. If no value is specified, the default values in [config](/src/config.py) will be used.
- `-b` can be used to set the batch size of the networks to train. Like above, the number of values given must be the same as the number of nets to train, and the order counts. If no value is specified, the default values in [config](/src/config.py) will be used.
- `-p` can be used to set the directory where to store the outputs of the training/prediction. If `-p` is specified, then such folder will be `../submissions/{-p}` rather than the default one `../submissions/submission_{timestamp_of_run}`.
- `-t` can be used to set the treshold to be used to create the submission file.
- `-s` can be used to use the run only to generate a prediction rather than training all the networks. To do so, `-s` should be set to `'predict'`, `-p` must be set to a output directory of a previous run and `-n` should be used to list the networks that the prediction ensemble should comprehend (if `-n` is not set, the ensemble will be generated out of the default six models). Obviously, to not run into an error when using `-s` to predict, the directory set in `-p` must contain the outputs of a previous training of the nets specified in `-n`. If you want to train the networks only, set `-s` to `train`.

Once the job created by the run above is finished, the project will be completed and all the outputs of the training, along with the final predictions, will be at the `submission_root` path set in [config](/src/config.py) or in the directory named by the parameter `-p` of the call to `trainer.py`.