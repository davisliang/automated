#!/bin/bash
echo "Starting up TweetFeeder."

STORMPATH="$HOME/apache-storm-0.9.5/lib/*"
TWITTERPATH="$HOME/twitter4j-4.0.4/lib/*"
CLASSPATH="$HOME/tweetnet/src/storm/"

javac -cp $STORMPATH:$TWITTERPATH ~/tweetnet/src/storm/TwitterStreamSpout.java ~/tweetnet/src/storm/TwitterCleanerBolt.java ~/tweetnet/src/storm/TwitterStorm.java

java -cp $STORMPATH:$TWITTERPATH:$CLASSPATH TwitterStorm #append_twitter_credentials_here
