package edu.jhu.thrax.hadoop.comparators;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.io.WritableUtils;

/**
 * Compares two primitive array objects lexicographically, except the zero-length array should be
 * sorted before any other.
 */
public class PrimitiveArrayMarginalComparator extends WritableComparator {
  
  public static final int[] MARGINAL = new int[0];
  
  public PrimitiveArrayMarginalComparator() {
    super(Text.class);
  }

  public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
    int h1 = WritableUtils.decodeVIntSize(b1[s1]);
    int length1 = (h1 == 1 ? b1[s1] : -1);
    
    int h2 = WritableUtils.decodeVIntSize(b2[s2]);
    int length2 = (h2 == 1 ? b2[s2] : -1);

    if (length1 == 0 && length2 == 0) return 0;
    if (length1 == 0) return -1;
    if (length2 == 0) return 1;
    return WritableComparator.compareBytes(b1, s1 + h1, l1 - h1, b2, s2 + h2, l2 - h2);
  }
}
