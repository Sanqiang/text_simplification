package edu.jhu.thrax.util;

import java.util.Map;
import java.util.regex.Pattern;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;

import edu.jhu.thrax.hadoop.datatypes.AlignmentWritable;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;

public class FormatUtils {

  /**
   * Field delimiter.
   */
  private static final String DELIMITER = "|||";

  /**
   * Regular expression for the field delimiter.
   */
  private static final String DELIMITER_REGEX = " \\|\\|\\| ";

  public static final String DELIM = String.format(" %s ", DELIMITER);

  public static final Pattern P_DELIM = Pattern.compile(DELIMITER_REGEX);
  public static final Pattern P_SPACE = Pattern.compile("\\s+");
  public static final Pattern P_EQUAL = Pattern.compile("=");
  public static final Pattern P_SLASH = Pattern.compile("/");
  public static final Pattern P_BSLASH = Pattern.compile("\\\\");
  public static final Pattern P_DASH = Pattern.compile("-");
  public static final Pattern P_COMMA_OR_SPACE = Pattern.compile("\\s+|,");


  public static boolean isNonterminal(String token) {
    return (token.charAt(0) == '[') && (token.charAt(token.length() - 1) == ']');
  }

  public static String stripNonterminal(String nt) {
    return nt.substring(1, nt.length() - 1);
  }

  public static String stripIndexedNonterminal(String nt) {
    return nt.substring(1, nt.length() - 3);
  }

  public static int getNonterminalIndex(String nt) {
    return Integer.parseInt(nt.substring(nt.length() - 2, nt.length() - 1));
  }

  public static String markup(String nt) {
    return "[" + nt + "]";
  }

  public static String markup(String nt, int index) {
    return nt.substring(0, nt.length() - 1) + "," + index + "]";
  }

  public static boolean isMonotonic(String input) {
    int one_pos = input.indexOf(",1]");
    int two_pos = input.indexOf(",2]");
    if (two_pos == -1 || one_pos == -1) return true;
    return (one_pos < two_pos);
  }

  public static String applyIndices(String input, boolean monotonic) {
    int first_nt = input.indexOf(",0]");
    if (first_nt == -1) return input;

    StringBuilder output = new StringBuilder(input);
    int second_nt = input.indexOf(",0]", first_nt + 1);
    if (second_nt == -1)
      output.setCharAt(first_nt + 1, '1');
    else {
      output.setCharAt(first_nt + 1, (monotonic ? '1' : '2'));
      output.setCharAt(second_nt + 1, (monotonic ? '2' : '1'));
    }
    return output.toString();
  }

  public static int[] applyIndices(int[] input, boolean monotonic) {
    boolean seen_first = false;
    for (int i = 0; i < input.length; ++i) {
      if (input[i] == 0) {
        if (seen_first) {
          input[i] = (monotonic ? -2 : -1);
        } else {
          input[i] = (monotonic ? -1 : -2);
          seen_first = true;
        }
      }
    }
    return input;
  }

  // TODO: this isn't a good place for this method any more.
  public static Text ruleToText(RuleWritable r, Map<String, Writable> fs, boolean label,
      boolean sparse) {
    if (r == null) throw new IllegalArgumentException("Cannot convert a null rule to text.");
    String alignment = null;    
    StringBuilder sb = new StringBuilder();
    sb.append(Vocabulary.word(r.lhs));
    sb.append(DELIM);
    int n = 1;
    for (int i = 0; i < r.source.length; ++i) {
      if (i != 0) sb.append(" ");
      if (Vocabulary.nt(r.source[i]))
        sb.append(markup(Vocabulary.word(r.source[i]), n++));
      else
        sb.append(Vocabulary.word(r.source[i]));
    }
    sb.append(DELIM);
    n = (r.monotone ? 1 : 2);
    for (int i = 0; i < r.target.length; ++i) {
      if (i != 0) sb.append(" ");
      if (Vocabulary.nt(r.target[i]))
        sb.append(markup(Vocabulary.word(r.target[i]), (r.monotone ? n++ : n--)));
      else
        sb.append(Vocabulary.word(r.target[i]));
    }

    sb.append(DELIM);
    for (String t : fs.keySet()) {
      String score;
      Writable val = fs.get(t);
      if (val instanceof FloatWritable) {
        float value = ((FloatWritable) fs.get(t)).get();
        if (value == -0.0 || Math.abs(value) < 0.000005)
          score = "0";
        else
          score = String.format("%.5f", value);
        if (sparse && Float.parseFloat(score) == 0) continue;
      } else if (val instanceof IntWritable) {
        score = String.format("%d", ((IntWritable) fs.get(t)).get());
        if (sparse && Integer.parseInt(score) == 0) continue;
      } else if (val instanceof Text) {
        score = ((Text) fs.get(t)).toString();
      } else if (val instanceof AlignmentWritable) {
        alignment = ((AlignmentWritable) val).toString(" ");
        continue;
      } else {
        throw new RuntimeException("Expecting float, integer, or string feature values.");
      }
      if (label)
        sb.append(String.format("%s=%s ", t, score));
      else
        sb.append(String.format("%s ", score));
    }
    if (alignment != null)
      sb.append(DELIMITER + " ").append(alignment + " ");
    return new Text(sb.substring(0, sb.length() - 1));
  }

  public static Text contextPhraseToText(Text phrase, Map<Text, Integer> fs) {
    if (phrase == null)
      throw new IllegalArgumentException("Cannot convert a null " + "phrase to Text.");
    StringBuilder sb = new StringBuilder();
    sb.append(phrase.toString());
    sb.append(DELIM);
    for (Text t : fs.keySet()) {
      int i = fs.get(t);
      if (i != 0) sb.append(String.format("%s=%d ", t, i));
    }
    return new Text(sb.substring(0, sb.length() - 1));
  }
}
