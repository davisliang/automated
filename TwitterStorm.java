
import java.util.*;
import backtype.storm.tuple.Fields;
import backtype.storm.tuple.Values;
import backtype.storm.Config;
import backtype.storm.LocalCluster;
import backtype.storm.topology.TopologyBuilder;

/**
 * Main class for storm topology. 
 */
public class TwitterStorm {

    /**
     * The main method extracts user arguments (in runAPI.sh), and constructs
     * the topology
     *
     * @param args[] array of size 5. Last argument are 'keyword' arguments
     */
    public static void main(String[] args) throws Exception{

        //grab authentication tokens
        String consumerKey = args[0];
        String consumerSecret = args[1];
        String accessToken = args[2];
        String accessTokenSecret = args[3];

        //grab keyword tokens
        String[] arguments = args.clone();
        String[] keyWords = Arrays.copyOfRange(arguments, 4, arguments.length);

        //create a new Storm configuration. 
        Config config = new Config();
        config.setDebug(true);

        //create a new topology. 
        TopologyBuilder builder = new TopologyBuilder();

        builder.setSpout("streamSpout", new TwitterStreamSpout(
            consumerKey,consumerSecret, accessToken, accessTokenSecret, keyWords));

        builder.setBolt("cleanerBolt", new TwitterCleanerBolt())
            .shuffleGrouping("streamSpout");
            
        //submit topology to local cluster.
        LocalCluster cluster = new LocalCluster();
        cluster.submitTopology("TwitterHashtagStorm", config,
            builder.createTopology());

        //no kill condition. Run until manual kill command.
    }
}
