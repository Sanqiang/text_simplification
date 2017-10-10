package edu.jhu.thrax.datatypes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;

import edu.jhu.thrax.extraction.SpanLabeler;
import edu.jhu.thrax.util.Vocabulary;

public class HierarchicalRule {
  private final PhrasePair lhs;
  private final PhrasePair[] nts;

  private final static PhrasePair[] EMPTY_NT_ARRAY = new PhrasePair[0];

  public HierarchicalRule(PhrasePair left_hand_side, PhrasePair[] nonterms) {
    lhs = left_hand_side;
    nts = nonterms;
  }

  public HierarchicalRule(PhrasePair leftHandSide) {
    this(leftHandSide, EMPTY_NT_ARRAY);
  }

  public int arity() {
    return nts.length;
  }

  public int numSourceTerminals() {
    int result = lhs.sourceLength();
    for (PhrasePair pp : nts)
      result -= pp.sourceLength();
    return result;
  }

  public int numTargetTerminals() {
    int result = lhs.targetLength();
    for (PhrasePair pp : nts)
      result -= pp.targetLength();
    return result;
  }

  public int numAlignmentPoints(Alignment a) {
    int result = lhs.numAlignmentPoints(a);
    for (PhrasePair pp : nts)
      result -= pp.numAlignmentPoints(a);
    return result;
  }

  public PhrasePair getLhs() {
    return lhs;
  }

  public PhrasePair getNonterminal(int index) {
    if (index < 0 || index > nts.length) return null;
    return nts[index];
  }

  public HierarchicalRule addNonterminal(PhrasePair pp) {
    PhrasePair[] theNTs = new PhrasePair[nts.length + 1];
    for (int i = 0; i < nts.length; i++)
      theNTs[i] = nts[i];
    theNTs[nts.length] = pp;
    return new HierarchicalRule(lhs, theNTs);
  }

  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append("HierarchicalRule { ");
    sb.append(String.format("lhs:%s ", lhs));
    for (int i = 0; i < nts.length; i++)
      sb.append(String.format("%d:%s ", i, nts[i]));
    sb.append("}");
    return sb.toString();
  }

  public String toString(int[] source, int[] target, SpanLabeler labeler, boolean source_labels) {
    return Vocabulary.word(lhsLabel(labeler, source_labels)) + " ||| "
        + Vocabulary.getWords(sourceSide(source, labeler, source_labels)) + " ||| "
        + Vocabulary.getWords(targetSide(target, labeler, source_labels));
  }

  public boolean equals(Object o) {
    if (o == this) return true;
    if (!(o instanceof HierarchicalRule)) return false;
    HierarchicalRule other = (HierarchicalRule) o;
    return lhs.equals(other.lhs) && Arrays.equals(nts, other.nts);
  }

  public int hashCode() {
    int result = 137;
    result = result * 37 + lhs.hashCode();
    result = result * 37 + Arrays.hashCode(nts);
    return result;
  }

  public int lhsLabel(SpanLabeler labeler, boolean use_source) {
    return lhs.getLabel(labeler, use_source);
  }

  private int ntLabel(int i, SpanLabeler labeler, boolean use_source) {
    if (i < 0 || i >= nts.length) return 0;
    return nts[i].getLabel(labeler, use_source);
  }

  public int[] sourceSide(int[] source, SpanLabeler labeler, boolean use_source) {
    int[] result = new int[numSourceTerminals() + nts.length];
    int n = 0, j = 0;
    for (int i = lhs.sourceStart; i < lhs.sourceEnd; ++i) {
      if (n < nts.length && i == nts[n].sourceStart) {
        result[j] = ntLabel(n, labeler, use_source);
        i = nts[n].sourceEnd - 1;
        ++n;
      } else {
        result[j] = source[i];
      }
      ++j;
    }
    return result;
  }

  public int[] targetSide(int[] target, SpanLabeler labeler, boolean use_source) {
    int[] result = new int[numTargetTerminals() + nts.length];
    int j = 0;
    boolean nt;
    for (int i = lhs.targetStart; i < lhs.targetEnd; ++i) {
      nt = false;
      for (int n = 0; n < nts.length; ++n) {
        if (i == nts[n].targetStart) {
          result[j] = ntLabel(n, labeler, use_source);
          i = nts[n].targetEnd - 1;
          nt = true;
          break;
        }
      }
      if (!nt) result[j] = target[i];
      ++j;
    }
    return result;
  }

  private int[] sourceToRule() {
    int[] result = new int[lhs.sourceEnd - lhs.sourceStart];
    int current = 0;
    int n = 0;
    for (int i = lhs.sourceStart; i < lhs.sourceEnd; i++) {
      if (n < nts.length && i == nts[n].sourceStart) {
        i = nts[n].sourceEnd - 1;
        ++n;
      } else {
        result[i - lhs.sourceStart] = current;
      }
      current++;
    }
    return result;
  }

  private int[] targetToRule() {
    int[] result = new int[lhs.targetEnd - lhs.targetStart];
    int current = 0;
    boolean nt;
    for (int i = lhs.targetStart; i < lhs.targetEnd; i++) {
      nt = false;
      for (int j = 0; j < arity(); j++) {
        if (i == nts[j].targetStart) {
          i = nts[j].targetEnd - 1;
          nt = true;
          break;
        }
      }
      if (!nt) result[i - lhs.targetStart] = current;
      current++;
    }
    return result;
  }

  public byte[] compactSourceAlignment(Alignment a) {
    int[] src_to_rule = sourceToRule();
    int[] tgt_to_rule = targetToRule();

    ArrayList<Byte> points = new ArrayList<Byte>();
    int n = 0;
    for (int i = lhs.sourceStart; i < lhs.sourceEnd; ++i) {
      if (n < nts.length && i == nts[n].sourceStart) {
        i = nts[n++].sourceEnd - 1;
      } else {
        Iterator<Integer> aligned = a.targetIndicesAlignedTo(i);
        while (aligned.hasNext()) {
          points.add((byte) src_to_rule[i - lhs.sourceStart]);
          points.add((byte) tgt_to_rule[aligned.next() - lhs.targetStart]);
        }
      }
    }
    byte[] result = new byte[points.size()];
    int i = 0;
    for (byte b : points)
      result[i++] = b;
    return result;
  }

  public byte[] compactTargetAlignment(Alignment a) {
    int[] tgt_to_rule = targetToRule();
    int[] src_to_rule = sourceToRule();
    ArrayList<Byte> points = new ArrayList<Byte>();
    boolean nt;
    for (int i = lhs.targetStart; i < lhs.targetEnd; ++i) {
      nt = false;
      for (int n = 0; n < nts.length; ++n) {
        if (i == nts[n].targetStart) {
          i = nts[n].targetEnd - 1;
          nt = true;
          break;
        }
      }
      if (!nt) {
        Iterator<Integer> aligned = a.sourceIndicesAlignedTo(i);
        while (aligned.hasNext()) {
          points.add((byte) tgt_to_rule[i - lhs.targetStart]);
          points.add((byte) src_to_rule[aligned.next() - lhs.sourceStart]);
        }
      }
    }
    byte[] result = new byte[points.size()];
    int i = 0;
    for (byte b : points)
      result[i++] = b;
    return result;
  }

  public boolean monotonic() {
    if (nts.length < 2) return true;
    return (nts[0].targetEnd <= nts[1].targetStart);
  }
}
