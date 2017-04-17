# Automated Multi-task Learning
**Automated MTL** supports two generalized multi-tasking, and recurrent deep learning architectures. Automated MTL uses the statistical regularities within the original dataset itself to reinforce the representations learned for the primary task. Automated MTL comes in two flavors: the CRNN (Cascaded Recurrent Neural Network) and the MRNN (Multi-tasking Recurrent Neural Network). 

The automated MTL architectures have achieved state-of-the-art performance in sentiment analysis, topic prediction, and hashtag recommendation using a diverse set of text corpuses including Twitter, Rotten Tomatoes, and IMDB.

A side project of automated MTL resulted in the ***Infinite Data Pipeline*** which is built on Java, Apache Storm, Kafka, and the Twitter API. The Infinite Data Pipeline streams and preprocesses Twitter data online and directly injects the streamed data into a running Tensorflow topology. 

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
1. [Install CUDA and cuDNN](http://tleyden.github.io/blog/2015/11/22/cuda-7-dot-5-on-aws-gpu-instance-running-ubuntu-14-dot-04/)
2. [Apache Storm and Twitter API Setup](https://www.tutorialspoint.com/apache_storm/apache_storm_installation.htm)
3. [Install keras and Theano](http://www.pyimagesearch.com/2016/07/18/installing-keras-for-deep-learning/)
4. [Download Kafka 2.10](https://www.apache.org/dyn/closer.cgi?path=/kafka/0.10.1.1/kafka_2.10-0.10.1.1.tgz)

## Data Miner Run Guide (MacOSX Local):
1. Run **systemStartMac.sh** to start your *Storm* instance. Make sure `KAFKAHOME` is set correctly in `scripts/startKafkaServer.sh`.
2. Edit `src/storm/pom.xml` with the appropriate Twitter credentials. Run `mvn install` inside `src/storm` to compile and `mvn exec:java` to start the data collection and streaming.

## Data Miner Run Guide (Ubuntu 16.04 Local):
1. Run **systemStartUbuntu.sh** to start your *Storm* instance. 
2. Run **runAPI.sh** to open the *Twitter* stream and start collection. (Requires you to edit **runAPI.sh** with correct *Twitter* API credentials).

## Tweetnet Run Guide:
1. Run **tweetnet.py**.

## Notes:

**Note**: The system start script opens five new terminals; *Apache Zookeeper*, the *Nimbus*, the *Supervisor*, *StormUI*, and the *Kafka* server. Each new open terminal requires **sudo** access and will request for the user's password. To view *StormUI* you can navigate to *localhost:8080*. 

**Note**: In the CUDA setup, the section where you link cuda to cuda-7.5 is outdated. 

Intead of following this step:

    export CUDA_HOME=/usr/local/cuda-7.5

Make sure you using and linking *CUDA v8.0*:

    export CUDA_HOME=/usr/local/cuda-8.0

**Note**: You will need to register for Twitter Developer credentials to run the data miner.
