package edu.jhu.thrax.tools;

import java.io.BufferedWriter;
import java.util.logging.Logger;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;

import edu.jhu.jerboa.util.FileManager;

public class SequenceToGrammar {

  private static final Logger logger = Logger.getLogger(SequenceToGrammar.class.getName());

  private static void usage() {
    System.err.println("Usage: java edu.jhu.thrax.tools.SequenceToGrammar");
    System.err.println("\t -i sequence_file \t Sequence file from Thrax grammar extraction.");
    System.err.println("\t -o output_file   \t Output grammar file name.");
    System.err.println();
  }

  public static void main(String[] args) throws Exception {
    String input_file = null;
    String output_file = null;

    if (args.length < 4 || args[0].toLowerCase().equals("-h")) {
      usage();
      System.exit(0);
    }
    for (int i = 0; i < args.length; i++) {
      if ("-i".equals(args[i]) && (i < args.length - 1)) {
        input_file = args[++i];
      } else if ("-o".equals(args[i]) && (i < args.length - 1)) {
        output_file = args[++i];
      }
    }
    if (input_file == null) {
      logger.severe("No input file specified.");
      usage();
      System.exit(0);
    }
    if (output_file == null) {
      logger.severe("No output file specified.");
      usage();
      System.exit(0);
    }

    Text rule_string = new Text();
    Configuration config = new Configuration();
    Path path = new Path(input_file);
    SequenceFile.Reader reader = new SequenceFile.Reader(FileSystem.getLocal(config), path, config);

    BufferedWriter grammar_writer = FileManager.getWriter(output_file);
    long rule_count = 0;
    while (reader.next(rule_string)) {
      grammar_writer.write(rule_string.toString());
      grammar_writer.newLine();
      rule_count++;
    }
    reader.close();
    grammar_writer.close();
    System.err.println("Merged " + rule_count + " rules.");
  }
}
