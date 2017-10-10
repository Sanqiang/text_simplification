package edu.jhu.thrax.util.io;

import java.util.ArrayList;

import edu.jhu.thrax.datatypes.AlignedSentencePair;
import edu.jhu.thrax.datatypes.Alignment;
import edu.jhu.thrax.datatypes.ArrayAlignment;
import edu.jhu.thrax.util.FormatUtils;
import edu.jhu.thrax.util.Vocabulary;
import edu.jhu.thrax.util.exceptions.MalformedInputException;
import edu.jhu.thrax.util.exceptions.MalformedParseException;

/**
 * Methods for validating user input. These should be used anywhere user input is received.
 */
public class InputUtilities {
  /**
   * Returns an array of the leaves of a parse tree, reading left to right.
   * 
   * @param parse a representation of a parse tree (Penn Treebank style)
   * @return an array of String giving the labels of the tree's leaves
   * @throws MalformedInputException if the parse tree is not well-formed
   */
  public static int[] parseYield(String input) throws MalformedInputException {
    if (input == null || input.isEmpty()) return new int[0];
    if (input.charAt(0) != '(') throw new MalformedInputException("malformed parse");
    
    ArrayList<Integer> words = new ArrayList<Integer>();
    
    int from = 0, to = 0;
    boolean seeking = true;
    boolean nonterminal = false;
    char current;
    // Run through entire (potentially parsed) sentence.
    while (from < input.length() && to < input.length()) {
      if (seeking) {
        // Seeking mode: looking for the start of the next symbol.
        current = input.charAt(from);
        // We skip brackets and spaces.
        if (current == '(' || current == ')' || current == ' ') {
          ++from;
          // Found a non spacing symbol, go into word filling mode.
        } else {
          to = from + 1;
          seeking = false;
          nonterminal = (input.charAt(from - 1) == '(');
        }
      } else {
        // Word filling mode. Advance to until we hit the end or spacing.
        current = input.charAt(to);
        if (current == ' ' || current == ')' || current == '(') {
          // Word ended.
            if (!nonterminal)
              words.add(Vocabulary.id(input.substring(from, to)));
          from = to + 1;
          seeking = true;
        } else {
          ++to;
        }
      }
    }
    int[] result = new int[words.size()];
    for (int i = 0; i < result.length; ++i)
      result[i] = words.get(i);
    return result;
  }

  /**
   * Returns the words (terminal symbols) represented by this input. If the input is a plain string,
   * returns whitespace-delimited tokens. If the input is a parse tree, returns an array of its
   * leaves.
   * 
   * @param input an input string
   * @param parsed whether the string represent a parse tree or not
   * @return an array of the terminal symbols represented by this input
   * @throws MalformedParseException if the input is a malformed parse tree and parsed is true
   */
  public static int[] getWords(String input, boolean parsed) throws MalformedInputException {
    String trimmed = input.trim();
    if (trimmed.isEmpty()) return new int[0];
    if (parsed) return parseYield(trimmed);
    return Vocabulary.addAll(trimmed);
  }

  public static AlignedSentencePair alignedSentencePair(String source, boolean source_is_parsed,
      String target, boolean target_is_parsed, String al, boolean reverse)
      throws MalformedInputException {
    int[] source_words = getWords(source, source_is_parsed);
    int[] target_words = getWords(target, target_is_parsed);
    if (source_words.length == 0 || target_words.length == 0)
      throw new MalformedInputException("empty sentence");
    Alignment alignment = ArrayAlignment.fromString(al.trim(), reverse);
    if (reverse) {
      if (!alignment.consistentWith(target_words.length, source_words.length))
        throw new MalformedInputException("inconsistent alignment");
      return new AlignedSentencePair(target_words, source_words, alignment);
    } else {
      if (!alignment.consistentWith(source_words.length, target_words.length))
        throw new MalformedInputException("inconsistent alignment");
      return new AlignedSentencePair(source_words, target_words, alignment);
    }
  }

  public static AlignedSentencePair alignedSentencePair(String line, boolean source_is_parsed,
      boolean target_is_parsed, boolean reverse) throws MalformedInputException {
    String[] parts = FormatUtils.P_DELIM.split(line);
    if (parts.length < 3) throw new MalformedInputException("not enough fields");
    return alignedSentencePair(parts[0], source_is_parsed, parts[1], target_is_parsed, parts[2],
        reverse);
  }
}
