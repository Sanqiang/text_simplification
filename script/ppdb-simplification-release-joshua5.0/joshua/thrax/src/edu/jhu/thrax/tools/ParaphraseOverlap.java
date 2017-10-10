package edu.jhu.thrax.tools;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.logging.Logger;

import edu.jhu.jerboa.util.FileManager;
import edu.jhu.thrax.util.FormatUtils;
import edu.jhu.thrax.util.io.LineReader;

public class ParaphraseOverlap {

  private static final Logger logger = Logger.getLogger(ParaphraseOverlap.class.getName());

  public static void main(String[] args) {

    String grammar_file = null;
    String reference_file = null;
    String weight_file = null;
    String output_file = null;

    for (int i = 0; i < args.length; i++) {
      if ("-g".equals(args[i]) && (i < args.length - 1)) {
        grammar_file = args[++i];
      } else if ("-r".equals(args[i]) && (i < args.length - 1)) {
        reference_file = args[++i];
      } else if ("-w".equals(args[i]) && (i < args.length - 1)) {
        weight_file = args[++i];
      } else if ("-o".equals(args[i]) && (i < args.length - 1)) {
        output_file = args[++i];
      }
    }

    if (grammar_file == null) {
      logger.severe("No grammar specified.");
      return;
    }
    if (reference_file == null) {
      logger.severe("No reference file specified.");
      return;
    }
    if (weight_file == null) {
      logger.severe("No weight file specified.");
      return;
    }
    if (output_file == null) {
      logger.severe("No output file specified.");
      return;
    }

    HashMap<String, Double> rule_to_score = new HashMap<String, Double>();
    HashMap<String, Double> weights = new HashMap<String, Double>();

    try {
      LineReader reference_reader = new LineReader(reference_file);
      while (reference_reader.hasNext()) {
        String line = reference_reader.next().trim();
        rule_to_score.put(line, null);
      }
      reference_reader.close();
      int num_references = rule_to_score.keySet().size();

      LineReader weights_reader = new LineReader(weight_file);
      while (weights_reader.hasNext()) {
        String line = weights_reader.next().trim();
        if (line.isEmpty()) continue;
        String[] fields = FormatUtils.P_SPACE.split(line);
        weights.put(fields[0], Double.parseDouble(fields[1]));
      }
      weights_reader.close();

      ArrayList<Double> missed = new ArrayList<Double>();

      LineReader reader = new LineReader(grammar_file);
      System.err.print("[");
      int rule_count = 0;
      while (reader.hasNext()) {
        String rule_line = reader.next().trim();

        String[] fields = FormatUtils.P_DELIM.split(rule_line);
        String rule = fields[0] + " ||| " + fields[1] + " ||| " + fields[2];

        double score = 0;
        String[] features = FormatUtils.P_SPACE.split(fields[3]);
        for (String f : features) {
          String[] parts = FormatUtils.P_EQUAL.split(f);
          if (weights.containsKey(parts[0]))
            score += weights.get(parts[0]) * Double.parseDouble(parts[1]);
        }

        if (rule_to_score.containsKey(rule)) {
          if (++rule_count % 10000 == 0) System.err.print("-");

          if (rule_to_score.get(rule) == null)
            rule_to_score.put(rule, score);
          else
            rule_to_score.put(rule, Math.max(score, rule_to_score.get(rule)));
        } else {
          missed.add(score);
        }
      }
      System.err.println("]");
      reader.close();

      double[] matched = new double[rule_count];
      int i = 0;
      for (Double s : rule_to_score.values())
        if (s != null) matched[i++] = s;
      rule_to_score = null;

      i = 0;
      double[] unmatched = new double[missed.size()];
      for (double s : missed)
        unmatched[i++] = s;
      missed = null;

      int num_correct = matched.length;
      int num_paraphrases = matched.length + unmatched.length;

      System.err.println("References:  " + num_references);
      System.err.println("Matched:     " + num_correct);
      System.err.println("Unmatched:   " + (num_references - num_correct));
      System.err.println("Nonmatching: " + unmatched.length);

      Arrays.sort(matched);
      Arrays.sort(unmatched);

      int m = 0, u = 0;
      BufferedWriter score_writer = FileManager.getWriter(output_file);
      while (m < matched.length && u < unmatched.length) {
        if (matched[m] < unmatched[u]) {
          if (m % 200 == 0)
            score_writer.write(matched[m] + "\t" + (num_correct / (double) num_references) + "\t"
                + (num_correct / (double) num_paraphrases) + "\n");
          m++;
          num_correct--;
        } else {
          u++;
        }
        num_paraphrases--;
      }
      score_writer.close();
    } catch (IOException e) {
      logger.severe(e.getMessage());
    }
  }
}
