#!/bin/bash
~/tweetnet/scripts/startZK.sh
sleep 5
~/tweetnet/scripts/newTerminalMac.sh ~/tweetnet/scripts/startZKClient.sh
sleep 5
~/tweetnet/scripts/newTerminalMac.sh ~/tweetnet/scripts/startNimbus.sh
sleep 5
~/tweetnet/scripts/newTerminalMac.sh ~/tweetnet/scripts/startSupervisor.sh
sleep 5
~/tweetnet/scripts/newTerminalMac.sh ~/tweetnet/scripts/startStormUI.sh
