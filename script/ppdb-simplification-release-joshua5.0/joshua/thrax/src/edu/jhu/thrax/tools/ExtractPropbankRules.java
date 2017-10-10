package edu.jhu.thrax.tools;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.Queue;
import java.util.logging.Logger;

import edu.jhu.jerboa.util.FileManager;
import edu.jhu.thrax.hadoop.datatypes.PrimitiveUtils;
import edu.jhu.thrax.syntax.LatticeArray;
import edu.jhu.thrax.util.Vocabulary;
import edu.jhu.thrax.util.exceptions.MalformedParseException;
import edu.jhu.thrax.util.io.LineReader;

public class ExtractPropbankRules {

  private static final Logger logger = Logger.getLogger(ExtractPropbankRules.class.getName());

  private static Collection<String> getLabels(LatticeArray parse, Span span) {
    HashSet<String> label_strings = new HashSet<String>();
    Collection<Integer> labels = parse.getConstituentLabels(span.f, span.t);
    for (int label : labels)
      label_strings.add(Vocabulary.word(label).replaceAll("Arg", "").replaceAll("Rel", ""));
    labels = parse.getCcgLabels(span.f, span.t);
    for (int label : labels)
      label_strings.add(Vocabulary.word(label).replaceAll("Arg", "").replaceAll("Rel", ""));
    return label_strings;
  }

  public static void main(String[] args) {

    String prop_file = null;
    String predicate_file = null;
    String relation_file = null;

    for (int i = 0; i < args.length; i++) {
      if ("-i".equals(args[i]) && (i < args.length - 1)) {
        prop_file = args[++i];
      } else if ("-p".equals(args[i]) && (i < args.length - 1)) {
        predicate_file = args[++i];
      } else if ("-r".equals(args[i]) && (i < args.length - 1)) {
        relation_file = args[++i];
      }
    }

    if (prop_file == null) {
      logger.severe("No propbank-annotated tree file specified.");
      return;
    }
    if (relation_file == null) {
      logger.severe("No relation output file specified.");
      return;
    }
    if (predicate_file == null) {
      logger.severe("No predicate output file specified.");
      return;
    }

    try {
      BufferedWriter predicate_writer = FileManager.getWriter(predicate_file);
      BufferedWriter relation_writer = FileManager.getWriter(relation_file);

      int line_count = 0;
      LineReader prop_reader = new LineReader(prop_file);
      while (prop_reader.hasNext()) {
        line_count++;

        String line = prop_reader.next().trim();
        LatticeArray parse = new LatticeArray(line, "bottom");
        String[] words = parse.getTerminals();
        int len = words.length;

        ArrayList<Span> rel_spans = new ArrayList<Span>();
        ArrayList<Span> arg = new ArrayList<Span>();
        for (int i = 0; i < len; ++i) {
          for (int j = i + 1; j <= len; ++j) {
            for (int id : parse.getConstituentLabels(i, j)) {
              String label = Vocabulary.word(id);
              if (label.endsWith("Rel")) {
                rel_spans.add(new Span(i, j));
              } else if (label.endsWith("Arg")) {
                arg.add(new Span(i, j));
              }
            }
          }
        }
        // Merge rel spans.
        boolean continuous = true;
        if (rel_spans.isEmpty()) continue;
        Span rel = rel_spans.get(0);
        for (int i = 1; i < rel_spans.size(); ++i)
          if (rel.suffix(rel_spans.get(i))) rel.t = rel_spans.get(i).t;

        // Build predicate-only entries.
        String rel_string = parse.getTerminalPhrase(rel.f, rel.t).toLowerCase();

        if (continuous) {
          for (String label : getLabels(parse, rel))
            predicate_writer.write(line_count + " ||| [" + label + "] ||| " + rel_string + "\n");
        }

        // Build full relation entries.
        if (!arg.isEmpty()) {
          int f = rel.f, t = rel.t;
          for (Span s : arg) {
            f = Math.min(f, s.f);
            t = Math.max(t, s.t);
          }
          int i = 0;
          int p = f;
          Queue<StringBuilder> surfaces = new LinkedList<StringBuilder>();
          surfaces.add(new StringBuilder());
          while (p < t) {
            if (i < arg.size() && p == arg.get(i).f) {
              int c = surfaces.size();
              while (c > 0) {
                StringBuilder b = surfaces.poll();
                for (String l : getLabels(parse, arg.get(i))) {
                  StringBuilder extension = new StringBuilder();
                  extension.append(b.toString());
                  extension.append("[" + l + "," + (i + 1) + "] ");
                  surfaces.add(extension);
                }
                c--;
              }
              p = arg.get(i).t;
              i++;
            } else {
              for (StringBuilder b : surfaces)
                b.append(parse.getTerminal(p).toLowerCase() + " ");
              p++;
            }
          }
          for (String l : getLabels(parse, new Span(f, t)))
            for (StringBuilder b : surfaces)
              relation_writer.write(line_count + " ||| [" + l + "] ||| "
                  + b.substring(0, b.length() - 1) + " ||| " + arg.size() + "\n");
        }
      }
      prop_reader.close();
      relation_writer.close();
      predicate_writer.close();
    } catch (IOException e) {
      logger.severe(e.getMessage());
    } catch (MalformedParseException e) {
      logger.severe(e.getMessage());
    }
  }
}


class Span implements Comparable<Span> {
  int f, t;

  public Span(int from, int to) {
    f = from;
    t = to;
  }

  @Override
  public int compareTo(Span that) {
    return PrimitiveUtils.compare(this.f, that.f);
  }

  public boolean suffix(Span that) {
    return this.t == that.f;
  }
}
