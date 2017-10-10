package edu.jhu.thrax.tools;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.PriorityQueue;
import java.util.logging.Logger;

import edu.jhu.jerboa.util.FileManager;
import edu.jhu.thrax.util.FormatUtils;
import edu.jhu.thrax.util.io.LineReader;

public class ParaphraseWordNet {

  private static final Logger logger = Logger.getLogger(ParaphraseWordNet.class.getName());

  public static void main(String[] args) {

    String grammar_file = null;
    String reference_file = null;
    String weight_file = null;
    String output_file = null;
    String relevant_file = null;

    for (int i = 0; i < args.length; i++) {
      if ("-g".equals(args[i]) && (i < args.length - 1)) {
        grammar_file = args[++i];
      } else if ("-r".equals(args[i]) && (i < args.length - 1)) {
        reference_file = args[++i];
      } else if ("-v".equals(args[i]) && (i < args.length - 1)) {
        relevant_file = args[++i];
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

    HashSet<String> reference_pairs = new HashSet<String>();
    HashSet<String> sources = new HashSet<String>();
    HashMap<String, Double> weights = new HashMap<String, Double>();
    try {
      LineReader reference_reader = new LineReader(reference_file);
      while (reference_reader.hasNext()) {
        String line = reference_reader.next().trim();
        String[] fields = FormatUtils.P_DELIM.split(line);
        reference_pairs.add(line);
        sources.add(fields[1]);
      }
      reference_reader.close();

      LineReader weights_reader = new LineReader(weight_file);
      while (weights_reader.hasNext()) {
        String line = weights_reader.next().trim();
        if (line.isEmpty()) continue;
        String[] fields = FormatUtils.P_SPACE.split(line);
        weights.put(fields[0], Double.parseDouble(fields[1]));
      }
      weights_reader.close();

      HashMap<String, Double> candidates = new HashMap<String, Double>();

      BufferedWriter rel_writer = null;
      if (relevant_file != null) rel_writer = FileManager.getWriter(relevant_file);

      LineReader reader = new LineReader(grammar_file);
      System.err.print("[");
      int count = 0;
      while (reader.hasNext()) {
        String rule_line = reader.next().trim();

        String[] fields = FormatUtils.P_DELIM.split(rule_line);

        if (!fields[0].startsWith("[VB") && !fields[0].startsWith("[NN")
            && !fields[0].startsWith("[JJ") && !fields[0].startsWith("[RB")
            && (!fields[1].contains(",1]"))) continue;

        String lhs = "[V]";
        if (fields[0].startsWith("[NN")) lhs = "[N]";
        if (fields[0].startsWith("[JJ")) lhs = "[A]";
        // Questionable tag, but the synsets don't seem to contain any adverbs anyway.
        if (fields[0].startsWith("[RB")) lhs = "[A]";

        String source = fields[1];
        String target = fields[2];

        if (!sources.contains(source)) continue;

        if (rel_writer != null) rel_writer.write(rule_line + "\n");

        double score = 0;
        String[] features = FormatUtils.P_SPACE.split(fields[3]);
        for (String f : features) {
          String[] parts = f.split("=");
          if (weights.containsKey(parts[0]))
            score += weights.get(parts[0]) * Double.parseDouble(parts[1]);
        }

        if (++count % 10000 == 0) System.err.print("-");

        String candidate = lhs + " ||| " + source + " ||| " + target;

        if (!candidates.containsKey(candidate)) {
          candidates.put(candidate, score);
        } else {
          double previous = candidates.get(candidate);
          candidates.put(candidate, Math.max(score, previous));
        }
      }
      System.err.println("]");
      reader.close();
      if (rel_writer != null) rel_writer.close();

      int num_paraphrases = 0;
      int num_correct = 0;
      int num_references = reference_pairs.size();

      PriorityQueue<ScoredEntry> entries = new PriorityQueue<ScoredEntry>();
      for (String p : candidates.keySet()) {
        if (reference_pairs.contains(p)) num_correct++;
        num_paraphrases++;
        entries.add(new ScoredEntry(p, candidates.get(p)));
      }

      System.err.println("References : " + num_references);
      System.err.println("Paraphrases: " + num_paraphrases);
      System.err.println("Correct:     " + num_correct);

      BufferedWriter score_writer = FileManager.getWriter(output_file);
      while (!entries.isEmpty()) {
        ScoredEntry e = entries.poll();
        if (reference_pairs.contains(e.pair)) {
          score_writer.write((num_correct / (double) num_references) + "\t"
              + (num_correct / (double) num_paraphrases) + "\n");
          num_correct--;
        }
        num_paraphrases--;
      }
      score_writer.close();
    } catch (IOException e) {
      logger.severe(e.getMessage());
    }
  }
}
