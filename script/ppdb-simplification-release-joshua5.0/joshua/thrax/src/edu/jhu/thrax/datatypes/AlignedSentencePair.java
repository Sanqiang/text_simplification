package edu.jhu.thrax.datatypes;

import java.util.Arrays;

public class AlignedSentencePair {
  public final int[] source;
  public final int[] target;
  public final Alignment alignment;

  public AlignedSentencePair(int[] ss, int[] ts, Alignment a) {
    source = ss;
    target = ts;
    alignment = a;
  }

  public boolean equals(Object o) {
    if (o == this) return true;
    if (!(o instanceof AlignedSentencePair)) return false;
    AlignedSentencePair other = (AlignedSentencePair) o;
    return Arrays.equals(source, other.source) && Arrays.equals(target, other.target)
        && alignment.equals(other.alignment);
  }

  public int hashCode() {
    int result = 137;
    result = result * 67 + Arrays.hashCode(source);
    result = result * 67 + Arrays.hashCode(target);
    result = result * 67 + alignment.hashCode();
    return result;
  }
}
