# tweetnet
Tweet Miner built on Java, Apache Storm, and the Twitter API. Requires twitter developer credentials to run.
Classifier (contextual LSTM) built on Keras (on Theano), and Python.

## Requirements:
1. CUDNN (tested on cuDNN 5105)
2. CUDA Drivers + NVIDIA Graphics Card with 5.0+ support (tested on GTX 1080)
3. Apache Zookeeper (tested on version 3.4.6)
4. Apache Storm (tested on version 0.9.5)
5. Twitter API + Developer Credentials (tested on version 4.0.4)
6. Theano (tested on version 0.8.2)
7. Keras (tested on latest version as of January 9, 2017)
8. Linux Based OS (tested on Ubuntu 16.04LTS)

## Install Guide:
1. [Install CUDA and cuDNN](http://www.pyimagesearch.com/2016/07/04/how-to-install-cuda-toolkit-and-cudnn-for-deep-learning/)
2. [Apache Storm and Twitter API Setup](https://www.tutorialspoint.com/apache_storm/apache_storm_in_twitter.htm)
3. [Install keras and Theano](http://www.pyimagesearch.com/2016/07/18/installing-keras-for-deep-learning/)



**Note**: In the CUDA setup, the section where you link cuda to cuda-7.5 is outdated. 

Intead of following this step:

    export CUDA_HOME=/usr/local/cuda-7.5

Make sure you using and linking *CUDA v8.0*:

    export CUDA_HOME=/usr/local/cuda-8.0
