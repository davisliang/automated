#!/bin/bash
sudo ~/tweetnet/scripts/startZK.sh
sleep 5
gnome-terminal -e ~/tweetnet/scripts/startZKClient.sh
sleep 5
gnome-terminal -e ~/tweetnet/scripts/startNimbus.sh
sleep 5
gnome-terminal -e ~/tweetnet/scripts/startSupervisor.sh
sleep 5
gnome-terminal -e ~/tweetnet/scripts/startStormUI.sh
