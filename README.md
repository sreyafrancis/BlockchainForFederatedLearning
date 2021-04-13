Decentralization of Artificial Intelligence with federated learning on Blockchain
====================================================================================

This repo contains the code and data for doing federated learning on MNIST dataset on Blockchain.

IEEE Paper - Record and reward federated learning contributions with blockchain
-----------------------------------------------------------------------------------

https://ieeexplore.ieee.org/document/8945913


Installation
------------

Before you do anything else you will first need to install the following python 
packages:

   `absl-py==0.5.0`
   `astor==0.7.1`
   `certifi==2018.10.15`
   `chardet==3.0.4`
   `Click==7.0`
   `cycler==0.10.0`
   `Flask==1.0.2`
   `gast==0.2.0`
   `grpcio==1.15.0`
   `h5py==2.8.0`
   `idna==2.7`
   `Jinja2==2.10`
   `Keras-Applications==1.0.6`
   `Keras-Preprocessing==1.0.5`
   `kiwisolver==1.0.1`
   `Markdown==3.0.1`
   `MarkupSafe==1.0`
   `matplotlib==3.0.0`
   `numpy==1.15.2`
   `protobuf==3.6.1`
   `pyparsing==2.2.2`
   `python-dateutil==2.7.3`
   `requests==2.20.0`
   `six==1.11.0`
   `tensorboard==1.11.0`
   `termcolor==1.1.0`
   `urllib3==1.24`
   `Werkzeug==0.14.1`
   `ItsDangerous==1.1.0`
   `tensorflow==1.11.0`
   
These are specified in the `src/packages_to_install.txt` file.
   
This project was built using Python3 but may work with Python2 given a few 
minor tweaks.

Preprocessing
-------------
The next step is to build the federated dataset to do federating learning on. You can prepare it by running this script:

    python src/data/federated_data_extractor.py
    
The default split is 2 `split_dataset(dataset,2)` which can be changed as per your number of clients.

Training
--------

Once you've generated chunks of `federated_data_x.d` you can begin training. For this simply 
run the following bash script:

    ./src/Run_BlockFL.sh

Assuming you've installed all `dependencies` and everything else successfully,
this should start federated learning on the generated federated datasets on blockchain.

Retrieving the models
----------------------

Once you've finished training, you can get the aggregated globally updated model  `federated_modelx.block` per round from the `src/blocks` folder.

    
 
