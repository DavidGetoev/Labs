import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import utils.ConfigurationUtil;

import java.io.IOException;


public class Main {

    private static final Logger log = LogManager.getLogger(Main.class.getName());

    public static void main( String[] args ) throws IOException {
        log.info(ConfigurationUtil.getConfigurationEntry("j"));
        //log.info(System.getProperty("j"));

    }
}
