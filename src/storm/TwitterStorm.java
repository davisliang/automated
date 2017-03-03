
import java.util.*;
import org.apache.storm.tuple.Fields;
import org.apache.storm.tuple.Values;
import org.apache.storm.Config;
import org.apache.storm.LocalCluster;
import org.apache.storm.topology.TopologyBuilder;
import org.apache.storm.kafka.bolt.KafkaBolt;
import org.apache.storm.kafka.bolt.selector.DefaultTopicSelector;
import org.apache.storm.kafka.bolt.mapper.*;
/**
 * Main class for storm topology.
 */


public class TwitterStorm {

    /**
     * The main method extracts user arguments (in runAPI.sh), and constructs
     * the topology. Optional Kill Command can be added at the end.
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

        TwitterStreamSpout streamSpout = new TwitterStreamSpout(
            consumerKey,consumerSecret, accessToken, accessTokenSecret, keyWords);

        // streamSpout.scheme = new SchemeAsMultiScheme(new KafkaBoltKeyValueScheme());
        builder.setSpout("streamSpout", streamSpout);

        TwitterCleanerBolt cleanerBolt = new TwitterCleanerBolt();

        builder.setBolt("cleanerBolt", cleanerBolt).shuffleGrouping("streamSpout");

        Properties props = new Properties();
        props.put("bootstrap.servers", "localhost:9092");
        props.put("acks", "1");
        props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
        props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");

        KafkaBolt kafkaBolt = new KafkaBolt()
                .withProducerProperties(props)
                .withTopicSelector(new DefaultTopicSelector("twitterstorm"))
                .withTupleToKafkaMapper(new FieldNameBasedTupleToKafkaMapper());

        builder.setBolt("forwardToKafka", kafkaBolt).shuffleGrouping("cleanerBolt");

        //submit topology to local cluster.
        LocalCluster cluster = new LocalCluster();
        cluster.submitTopology("TwitterHashtagStorm", config,
            builder.createTopology());
        //no kill condition. Run until manual kill command.
    }
}
