#!/bin/bash
echo "Starting up TweetFeeder."

STORMPATH="/Users/Davis/apache-storm-0.9.5/lib/*"
TWITTERPATH="/Users/Davis/twitter4j-4.0.4/lib/*"
CLASSPATH="/Users/Davis/tweetnet/src/"

javac -cp $STORMPATH:$TWITTERPATH ~/tweetnet/src/TwitterStreamSpout.java ~/tweetnet/src/TwitterCleanerBolt.java ~/tweetnet/src/TwitterStorm.java

java -cp $STORMPATH:$TWITTERPATH:$CLASSPATH TwitterStorm mneD3kPggHIGMF3uzGrOYP6YT vb66pHEGPRbadl1fpwN4Um8yO4or46IigKVnYrvfP1obPLMZ4k  306922539-8OcnG1IASwyAjceFWekGt4quMrGDcxNmMtcmJ7ON  RJtwaYKfHvKkeI8YZdPVZ4zJLn9FCkZ3OK6aPkCiteQPC 
