package edu.jhu.thrax.tools;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.logging.Logger;

import edu.jhu.jerboa.util.FileManager;
import edu.jhu.thrax.util.FormatUtils;
import edu.jhu.thrax.util.io.LineReader;

public class SplitAndFilter {

  private static final Logger logger = Logger.getLogger(SplitAndFilter.class.getName());

  @SuppressWarnings("unchecked")
  public static void main(String[] args) {

    String grammar_file = null;
    String filter_file = null;
    String output_prefix = null;

    for (int i = 0; i < args.length; i++) {
      if ("-g".equals(args[i]) && (i < args.length - 1)) {
        grammar_file = args[++i];
      } else if ("-f".equals(args[i]) && (i < args.length - 1)) {
        filter_file = args[++i];
      } else if ("-o".equals(args[i]) && (i < args.length - 1)) {
        output_prefix = args[++i];
      }
    }

    if (grammar_file == null) {
      logger.severe("No grammar specified.");
      return;
    }
    if (filter_file == null) {
      logger.severe("No filter file specified.");
      return;
    }
    if (output_prefix == null) {
      logger.severe("No output prefix specified.");
      return;
    }

    int lex_count = 0, phr_count = 0, syn_count = 0, drop_count = 0;

    HashSet<String> filter = new HashSet<String>();
    HashMap<String, Integer> stop_count = new HashMap<String, Integer>();
    try {
      LineReader filter_reader = new LineReader(filter_file);
      while (filter_reader.hasNext()) {
        String word = filter_reader.next().trim();
        filter.add(word);
        stop_count.put(word, 0);
      }
      filter_reader.close();
    } catch (IOException e) {
      logger.severe(e.getMessage());
    }

    try {
      LineReader reader = new LineReader(grammar_file);
      BufferedWriter lex_writer = FileManager.getWriter(output_prefix + ".lexical.gz");
      BufferedWriter phr_writer = FileManager.getWriter(output_prefix + ".phrasal.gz");
      BufferedWriter syn_writer = FileManager.getWriter(output_prefix + ".syntax.gz");

      BufferedWriter lex_self_writer = FileManager.getWriter(output_prefix + ".lexical-self.gz");
      BufferedWriter phr_self_writer = FileManager.getWriter(output_prefix + ".phrasal-self.gz");
      BufferedWriter syn_self_writer = FileManager.getWriter(output_prefix + ".syntax-self.gz");

      BufferedWriter stop_writer = FileManager.getWriter(output_prefix + ".stop.gz");
      BufferedWriter stats_writer = FileManager.getWriter(output_prefix + ".stats.txt");

      HashSet<String> source_words = new HashSet<String>();
      HashSet<String> target_words = new HashSet<String>();

      while (reader.hasNext()) {
        String rule_line = reader.next().trim();
        boolean phrasal = true;
        boolean drop = true;

        try {
          String[] fields = FormatUtils.P_DELIM.split(rule_line);
          String[] source = FormatUtils.P_SPACE.split(fields[1]);
          String[] target = FormatUtils.P_SPACE.split(fields[2]);

          boolean self = fields[1].equals(fields[2]);

          source_words.clear();
          target_words.clear();
          for (String word : source) {
            if (word.startsWith("["))
              phrasal = false;
            else
              source_words.add(word);
          }
          for (String word : target)
            if (!word.startsWith("[")) target_words.add(word);

          if (!self) {
            HashSet<String> source_added = (HashSet<String>) source_words.clone();
            HashSet<String> target_added = (HashSet<String>) target_words.clone();

            source_added.removeAll(target_words);
            target_added.removeAll(source_words);

            for (String word : source_added)
              if (!filter.contains(word))
                drop = false;
              else
                stop_count.put(word, stop_count.get(word) + 1);
            for (String word : target_added)
              if (!filter.contains(word))
                drop = false;
              else
                stop_count.put(word, stop_count.get(word) + 1);
          } else {
            drop = false;
          }

          // Dropped rule.
          if (drop) {
            stop_writer.write(rule_line);
            stop_writer.newLine();
            drop_count++;
            continue;
          }

          // Lexical rule.
          if (phrasal && source.length == 1 && target.length == 1) {
            if (self) {
              lex_self_writer.write(rule_line);
              lex_self_writer.newLine();
            } else {
              lex_writer.write(rule_line);
              lex_writer.newLine();
            }
            lex_count++;
            continue;
          }

          // Phrasal rule.
          if (phrasal) {
            if (self) {
              phr_self_writer.write(rule_line);
              phr_self_writer.newLine();
            } else {
              phr_writer.write(rule_line);
              phr_writer.newLine();
            }
            phr_count++;
            continue;
          }

          // Syntactic rule.
          if (self) {
            syn_self_writer.write(rule_line);
            syn_self_writer.newLine();
          } else {
            syn_writer.write(rule_line);
            syn_writer.newLine();
          }
          syn_count++;
        } catch (Exception e) {
          logger.warning(e.getMessage());
          logger.warning(rule_line);
          continue;
        }
      }
      reader.close();

      for (String word : stop_count.keySet())
        stats_writer.write(word + "\t" + stop_count.get(word) + "\n");

      System.err.println("Total:  \t" + (lex_count + phr_count + syn_count + drop_count));
      System.out.println("Dropped:\t" + drop_count);
      System.out.println("Lexical:\t" + lex_count);
      System.out.println("Phrasal:\t" + phr_count);
      System.out.println("Syntactic:\t" + syn_count);

      lex_writer.close();
      phr_writer.close();
      syn_writer.close();
      stop_writer.close();
      lex_self_writer.close();
      phr_self_writer.close();
      syn_self_writer.close();
      stats_writer.close();
    } catch (IOException e) {
      logger.severe(e.getMessage());
    }
  }

}
