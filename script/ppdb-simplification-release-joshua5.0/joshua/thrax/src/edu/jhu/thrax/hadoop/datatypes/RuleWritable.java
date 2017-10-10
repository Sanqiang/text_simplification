package edu.jhu.thrax.hadoop.datatypes;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.io.WritableUtils;
import org.apache.hadoop.mapreduce.Partitioner;

import edu.jhu.thrax.hadoop.comparators.FieldComparator;
import edu.jhu.thrax.hadoop.comparators.PrimitiveArrayMarginalComparator;
import edu.jhu.thrax.util.FormatUtils;
import edu.jhu.thrax.util.Vocabulary;

public class RuleWritable implements WritableComparable<RuleWritable> {

  public int lhs;
  public int[] source;
  public int[] target;
  public boolean monotone;

  public RuleWritable() {
    source = null;
    target = null;
  }

  // TODO: this is broken.
  @Deprecated
  public RuleWritable(String r) {
    String[] fields = FormatUtils.P_DELIM.split(r);
    lhs = Vocabulary.id(fields[0]);
    source = Vocabulary.addAll(fields[1]);
    target = Vocabulary.addAll(fields[2]);
    monotone = true;
  }

  public RuleWritable(RuleWritable r) {
    this.set(r);
  }

  public RuleWritable(int left, int[] src, int[] tgt, boolean m) {
    lhs = left;
    source = src;
    target = tgt;
    monotone = m;
  }

  public void set(RuleWritable r) {
    lhs = r.lhs;
    source = Arrays.copyOf(r.source, r.source.length);
    target = Arrays.copyOf(r.target, r.target.length);
    monotone = r.monotone;
  }

  // TODO: either encode monotone in LHS sign or always write LHS as positive (for marginal sort)
  public void write(DataOutput out) throws IOException {
    out.writeBoolean(monotone);
    WritableUtils.writeVInt(out, lhs);
    PrimitiveUtils.writeIntArray(out, source);
    PrimitiveUtils.writeIntArray(out, target);
  }

  public void readFields(DataInput in) throws IOException {
    monotone = in.readBoolean();
    lhs = WritableUtils.readVInt(in);
    source = PrimitiveUtils.readIntArray(in);
    target = PrimitiveUtils.readIntArray(in);
  }

  public boolean sameYield(RuleWritable r) {
    return monotone == r.monotone && lhs == r.lhs && Arrays.equals(source, r.source)
        && Arrays.equals(target, r.target);
  }

  public boolean equals(Object o) {
    if (o instanceof RuleWritable) return sameYield((RuleWritable) o);
    return false;
  }

  public int hashCode() {
    int result = 163;
    result = 37 * result + lhs;
    result = 37 * result + (monotone ? 1 : 0);
    result = 37 * result + Arrays.hashCode(source);
    result = 37 * result + Arrays.hashCode(target);
    return result;
  }

  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append(Vocabulary.word(lhs));
    sb.append(FormatUtils.DELIM);
    int n = 1;
    for (int i = 0; i < source.length; ++i) {
      if (i != 0) sb.append(" ");
      if (Vocabulary.nt(source[i]))
        sb.append(FormatUtils.markup(Vocabulary.word(source[i]), n++));
      else
        sb.append(Vocabulary.word(source[i]));
    }
    sb.append(FormatUtils.DELIM);
    n = (monotone ? 1 : 2);
    for (int i = 0; i < target.length; ++i) {
      if (i != 0) sb.append(" ");
      if (Vocabulary.nt(target[i]))
        sb.append(FormatUtils.markup(Vocabulary.word(target[i]), (monotone ? n++ : n--)));
      else
        sb.append(Vocabulary.word(target[i]));
    }
    return sb.toString();
  }

  public int compareTo(RuleWritable that) {
    int cmp = PrimitiveUtils.compareIntArrays(this.source, that.source);
    if (cmp != 0) return cmp;
    cmp = PrimitiveUtils.compare(this.lhs, that.lhs);
    if (cmp != 0) return cmp;
    cmp = PrimitiveUtils.compareIntArrays(this.target, that.target);
    if (cmp != 0) return cmp;
    cmp = PrimitiveUtils.compare(this.monotone, that.monotone);
    return cmp;
  }

  public static final int size(byte[] b, int s, int l) throws IOException {
    int pos = s;
    pos += WritableUtils.decodeVIntSize(b[s + 1]) + 1;
    
    int srclen_size = WritableUtils.decodeVIntSize(b[pos]);
    int source_size = WritableComparator.readVInt(b, pos);
    
    pos += srclen_size + source_size;
    
    int tgtlen_size = WritableUtils.decodeVIntSize(b[pos]);
    int target_size = WritableComparator.readVInt(b, pos);
    
    pos += tgtlen_size + target_size;
    
    return pos - s;
  }
  
  public static class YieldPartitioner extends Partitioner<RuleWritable, Writable> {
    public int getPartition(RuleWritable key, Writable value, int numPartitions) {
      return (key.hashCode() & Integer.MAX_VALUE) % numPartitions;
    }
  }
  
  public static class FirstWordPartitioner extends Partitioner<RuleWritable, Writable> {
    public int getPartition(RuleWritable key, Writable value, int numPartitions) {
      if (key.source.length == 0) return 0;
      return (key.source[0] & Integer.MAX_VALUE) % numPartitions;
    }
  }

  static {
    WritableComparator.define(RuleWritable.class, new YieldComparator());
  }

  public static class YieldComparator extends WritableComparator {
    private static final WritableComparator PARRAY_COMP = new PrimitiveArrayMarginalComparator();
    private static final FieldComparator SOURCE_COMP = new FieldComparator(0, PARRAY_COMP);
    private static final FieldComparator TARGET_COMP = new FieldComparator(1, PARRAY_COMP);

    public YieldComparator() {
      super(RuleWritable.class);
    }

    public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
      try {
        int h1 = WritableUtils.decodeVIntSize(b1[s1 + 1]) + 1;
        int h2 = WritableUtils.decodeVIntSize(b2[s2 + 1]) + 1;

        int cmp = SOURCE_COMP.compare(b1, s1 + h1, l1 - h1, b2, s2 + h2, l2 - h2);
        if (cmp != 0) return cmp;

        int lhs1 = WritableComparator.readVInt(b1, s1 + 1);
        int lhs2 = WritableComparator.readVInt(b2, s2 + 1);
        cmp = PrimitiveUtils.compare(lhs1, lhs2);
        if (cmp != 0) return cmp;

        cmp = TARGET_COMP.compare(b1, s1 + h1, l1 - h1, b2, s2 + h2, l2 - h2);
        if (cmp != 0) return cmp;

        // Comparing encoded monotone flag.
        return PrimitiveUtils.compare(b1[s1], b2[s2]);
      } catch (IOException e) {
        throw new IllegalArgumentException(e);
      }
    }
  }

  public static class LHSPartitioner extends Partitioner<RuleWritable, Writable> {
    public int getPartition(RuleWritable key, Writable value, int numPartitions) {
      return (key.lhs & Integer.MAX_VALUE) % numPartitions;
    }
  }

  public static class SourcePartitioner extends Partitioner<RuleWritable, Writable> {
    public int getPartition(RuleWritable key, Writable value, int numPartitions) {
      return (Arrays.hashCode(key.source) & Integer.MAX_VALUE) % numPartitions;
    }
  }

  // TODO: ensure that monotone doesn't need to be factored in here. (it does.)
  public static class TargetPartitioner extends Partitioner<RuleWritable, Writable> {
    public int getPartition(RuleWritable key, Writable value, int numPartitions) {
      return (Arrays.hashCode(key.target) & Integer.MAX_VALUE) % numPartitions;
    }
  }
}
