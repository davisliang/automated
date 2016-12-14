
import java.util.HashMap;
import java.util.Map;
import java.util.Arrays;
import twitter4j.*;
import twitter4j.conf.*;
import java.util.List;
import backtype.storm.tuple.Fields;
import backtype.storm.tuple.Values;

import backtype.storm.task.OutputCollector;
import backtype.storm.task.TopologyContext;
import backtype.storm.topology.IRichBolt;
import backtype.storm.topology.OutputFieldsDeclarer;
import backtype.storm.tuple.Tuple;

import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * this class parses and cleans the data fed in from a TwitterStreamSpout object.
 *
 **/
public class TwitterCleanerBolt implements IRichBolt {

	private OutputCollector collector;
	boolean useTopicSelector = false;
	String language = new String("en");
	List<String> topics = Arrays.asList("politics","entertainment",
			"world","us","business","opinion","tech","science","health", 
			"sports", "art", "style", "food", "travel");

	/**
	 * this method sets up the environment for the bolt on a node.
	 *
	 * @param conf  The storm config for the bolt. Provided to topology.
	 * @param context Used to get info about task's placement in topology (ID, I/O, etc.).
	 * @param collector Used to emit tuples from the bolt. Collector is thread-safe.
	 *
	 **/
	@Override
	public void prepare(Map conf, TopologyContext context, OutputCollector collector) {
	  this.collector = collector;
	}

	/**
	 * This method runs all of the parsing and cleanup using helper methods.
	 * See this link: http://twitter4j.org/javadoc/twitter4j/Status.html
	 *
	 * @param tuple A single tweet block with fields and metadata.
	 *
	 **/
	@Override
	public void execute(Tuple tuple) {

		Status tweet = (Status) tuple.getValueByField("tweet");
		
		// return criterion
		if(!tweet.getLang().equals(language))
		 	return;
		if(tweet.isRetweet())
		 	return;
		if(tweet.getHashtagEntities().length == 0)
			return;

		//remove URLs from tweets, removes newlines, and removes casing
		//if resulting tweet is less than 80 characters, return.
		String txt = tweet.getText();
		txt = this.removeUrl(txt);
		txt = txt.replace("\n", "");
		txt = txt.toLowerCase();
		if(txt.length()<80)
			return;
			
		// extract hashtags
		String hasht = "\nhashtags: ";
		boolean keep = false;
		for(HashtagEntity hashtage : tweet.getHashtagEntities()) {
			this.collector.emit(new Values(hashtage.getText())); 

			// only select tweets that have hashtags/topics co-occurrence. 
			if(this.useTopicSelector == true){

				for(String s:this.topics) {
					if(s.equals(hashtage.getText())) {
						hasht += hashtage.getText();
						keep = true;
					}
				}
			} else {
				hasht += hashtage.getText() + " ";
			}

			if(useTopicSelector == true && keep == false)
				return;

			OutputStream oStream;
			String finaltext = "\n\ntext: " + txt.replaceAll("#[^\\s]+","").replaceAll("@[^\\s]+","");

			try {
				oStream = new FileOutputStream("/Users/Davis/Desktop/dump.txt", true);
				oStream.write(finaltext.getBytes());
				oStream.write(hasht.getBytes());
				oStream.close();
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
    }

    /**
     * TO-DO
     **/
	@Override
	public void cleanup() {
    return;
  }

	/**
     * TO-DO: Documentation
     **/
	@Override
	public void declareOutputFields(OutputFieldsDeclarer declarer) {
		declarer.declare(new Fields("postCleanedTweets"));
	}

	/**
     * TO-DO
     **/
	@Override
	public Map<String, Object> getComponentConfiguration() {
		return null;
	}

	/**
	 * helper method to remove all URLs using regex.
	 * @param tweet The text field from the tweet tuple.
	 **/
	public static String removeUrl(String tweet) {
		try{
			String urlPattern = "((https?|ftp|gopher|telnet|file|Unsure|http):((//)|(\\\\))+[\\w\\d:#@%/;$()~_?\\+-=\\\\\\.&]*)";
			
			Pattern p = Pattern.compile(urlPattern,Pattern.CASE_INSENSITIVE);
			Matcher m = p.matcher(tweet);
			
			// re-assigns str while removing URLs
			int i = 0;
			while (m.find()) {
			    tweet = tweet.replaceAll(m.group(i),"").trim();
			    i++;
			}

			return tweet;

		} catch(Exception e) {
			return tweet;
		}
	}

} 
