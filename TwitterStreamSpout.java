 
import java.util.Map;
import java.util.concurrent.LinkedBlockingQueue;

import twitter4j.FilterQuery;
import twitter4j.StallWarning;
import twitter4j.Status;
import twitter4j.StatusDeletionNotice;
import twitter4j.StatusListener;

import twitter4j.TwitterStream;
import twitter4j.TwitterStreamFactory;
import twitter4j.auth.AccessToken;
import twitter4j.conf.ConfigurationBuilder;

import backtype.storm.Config;
import backtype.storm.spout.SpoutOutputCollector;

import backtype.storm.task.TopologyContext;
import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.topology.base.BaseRichSpout;
import backtype.storm.tuple.Fields;
import backtype.storm.tuple.Values;

import backtype.storm.utils.Utils;

/**
 * this class talks directly the the twitterAPI using the user credentials
 * in runAPI.sh. The data from this spout feeds into the parser and cleaner
 * bolts.
 */
@SuppressWarnings("serial")
public class TwitterStreamSpout extends BaseRichSpout {
    SpoutOutputCollector _collector;
    LinkedBlockingQueue<Status> queue = null;
    TwitterStream _twitterStream;
    
    String consumerKey;
    String consumerSecret;
    String accessToken;
    String accessTokenSecret;
    String[] keyWords;
        
    public TwitterStreamSpout(String consumerKey, String consumerSecret,
        String accessToken, String accessTokenSecret, String[] keyWords) {

        this.consumerKey = consumerKey;
        this.consumerSecret = consumerSecret;
        this.accessToken = accessToken;
        this.accessTokenSecret = accessTokenSecret;
        this.keyWords = keyWords;
    }
        
    public TwitterStreamSpout() {
      // TODO Auto-generated constructor stub
    }
        
    @Override
    public void open(Map conf, TopologyContext context,
        SpoutOutputCollector collector) {

        queue = new LinkedBlockingQueue<Status>(1000);
        _collector = collector;

        StatusListener listener = new StatusListener() {

        @Override
        public void onStatus(Status status) {
           queue.offer(status);
        }
                
        @Override
        public void onDeletionNotice(StatusDeletionNotice sdn) {}
                
        @Override
        public void onTrackLimitationNotice(int i) {}
                
        @Override
        public void onScrubGeo(long l, long l1) {}
                
        @Override
        public void onException(Exception ex) {}
                
        @Override
        public void onStallWarning(StallWarning arg0) {
           // TODO Auto-generated method stub
        }
    };
                
        ConfigurationBuilder cb = new ConfigurationBuilder(); 
            
        cb.setDebugEnabled(true)
        .setOAuthConsumerKey(consumerKey)
        .setOAuthConsumerSecret(consumerSecret)
        .setOAuthAccessToken(accessToken)
        .setOAuthAccessTokenSecret(accessTokenSecret);
                
        _twitterStream = new TwitterStreamFactory(cb.build()).getInstance();
        _twitterStream.addListener(listener);
            
        if (keyWords.length == 0) {
            _twitterStream.sample();
        }else {
            FilterQuery query = new FilterQuery().track(keyWords);
            _twitterStream.filter(query);
        }
    }
            
    @Override
    public void nextTuple() {
        Status ret = queue.poll();
                
        if (ret == null) {
            Utils.sleep(50);
        } else {
            _collector.emit(new Values(ret));
        }
    }
            
    @Override
    public void close() {
        _twitterStream.shutdown();
    }
            
    @Override
    public Map<String, Object> getComponentConfiguration() {
        Config ret = new Config();
        ret.setMaxTaskParallelism(1);
        return ret;
    }
            
    @Override
    public void ack(Object id) {}
            
    @Override
    public void fail(Object id) {}
            
    @Override
    public void declareOutputFields(OutputFieldsDeclarer declarer) {
        declarer.declare(new Fields("tweet"));
    }
}
