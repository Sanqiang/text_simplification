package edu.jhu.thrax.extraction;

import java.util.List;

import edu.jhu.thrax.syntax.ParseTree;
import edu.jhu.thrax.util.Vocabulary;

public class SAMTLabeler implements SpanLabeler {

  private boolean allowConstituent = true;
  private boolean allowCCG = true;
  private boolean allowConcat = true;
  private boolean allowDoubleConcat = true;
  private UnaryCategoryHandler unaryCategoryHandler;

  private ParseTree tree;
  private int defaultLabel;

  public SAMTLabeler(String parse, boolean constituent, boolean ccg, boolean concat,
      boolean doubleConcat, String unary, int def) {
    allowConstituent = constituent;
    allowCCG = ccg;
    allowConcat = concat;
    allowDoubleConcat = doubleConcat;
    defaultLabel = def;
    unaryCategoryHandler = UnaryCategoryHandler.fromString(unary);
    tree = ParseTree.fromPennFormat(parse);
    if (tree == null) System.err.printf("WARNING: SAMT labeler: %s is not a parse tree\n", parse);
  }

  public int getLabel(int from, int to) {
    if (tree == null) return defaultLabel;
    int label;
    if (allowConstituent) {
      label = constituentLabel(from, to);
      if (label != 0) return label;
    }
    if (allowConcat) {
      label = concatenatedLabel(from, to);
      if (label != 0) return label;
    }
    if (allowCCG) {
      label = forwardSlashLabel(from, to);
      if (label != 0) return label;
      label = backwardSlashLabel(from, to);
      if (label != 0) return label;
    }
    if (allowDoubleConcat) {
      label = doubleConcatenatedLabel(from, to);
      if (label != 0) return label;
    }
    return defaultLabel;
  }

  private int constituentLabel(int from, int to) {
    List<ParseTree.Node> nodes = tree.internalNodesWithSpan(from, to);
    if (nodes.isEmpty()) return 0;
    switch (unaryCategoryHandler) {
      case TOP:
        return nodes.get(0).label();
      case BOTTOM:
        return nodes.get(nodes.size() - 1).label();
      case ALL:
        // TODO: currently broken.
        String result = Vocabulary.word(nodes.get(0).label());
        for (int i = 1; i < nodes.size(); i++)
          result += ":" + Vocabulary.word(nodes.get(i).label());
        return Vocabulary.id(result);
    }
    return 0;
  }

  private int concatenatedLabel(int from, int to) {
    for (int mid = from + 1; mid < to; mid++) {
      int a = constituentLabel(from, mid);
      int b = constituentLabel(mid, to);
      if (a != 0 && b != 0) return LabelCache.PLUS.get(a, b);
    }
    return 0;
  }

  private int forwardSlashLabel(int from, int to) {
    for (int end = to + 1; end <= tree.numLeaves(); end++) {
      int a = constituentLabel(from, end);
      int b = constituentLabel(to, end);
      if (a != 0 && b != 0) return LabelCache.SLASH.get(a, b);
    }
    return 0;
  }

  private int backwardSlashLabel(int from, int to) {
    for (int start = from - 1; start >= 0; start--) {
      int a = constituentLabel(start, to);
      int b = constituentLabel(start, from);
      if (a != 0 && b != 0) return LabelCache.BACKSLASH.get(a, b);
    }
    return 0;
  }

  private int doubleConcatenatedLabel(int from, int to) {
    for (int mid1 = from + 1; mid1 < to - 1; mid1++) {
      for (int mid2 = mid1 + 1; mid2 < to; mid2++) {
        int a = constituentLabel(from, mid1);
        int b = constituentLabel(mid1, mid2);
        int c = constituentLabel(mid2, to);
        if (a != 0 && b != 0 && c != 0) return LabelCache.PLUS.get(LabelCache.PLUS.get(a, b), c);
      }
    }
    return 0;
  }

  private enum UnaryCategoryHandler {
    TOP, BOTTOM, ALL;

    public static UnaryCategoryHandler fromString(String s) {
      if (s.equalsIgnoreCase("top"))
        return TOP;
      else if (s.equalsIgnoreCase("bottom"))
        return BOTTOM;
      else
        return ALL;
    }
  }
}
