#!/bin/bash
echo "starting kafka client"

KAFKAHOME="$HOME/kafka-0.10.1.1-src"

# Added the sudo due to file permissions being messed up
sudo $KAFKAHOME/bin/kafka-server-start.sh $KAFKAHOME/config/server.properties
