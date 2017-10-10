package edu.jhu.thrax.hadoop.datatypes;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.mapreduce.Partitioner;

import edu.jhu.thrax.util.FormatUtils;

public class AlignedRuleWritable implements WritableComparable<AlignedRuleWritable> {

  private RuleWritable rule;
  private AlignmentWritable alignment;

  public AlignedRuleWritable() {
    rule = null;
    alignment = null;
  }

  public AlignedRuleWritable(RuleWritable r, AlignmentWritable a) {
    rule = r;
    alignment = a;
  }

  public RuleWritable getRule() {
    return rule;
  }

  public AlignmentWritable getAlignment() {
    return alignment;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    rule = new RuleWritable();
    alignment = new AlignmentWritable();
    
    rule.readFields(in);
    alignment.readFields(in);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    rule.write(out);
    alignment.write(out);
  }

  public boolean equals(Object o) {
    if (o instanceof AlignedRuleWritable)
      return rule.equals(((AlignedRuleWritable) o).rule)
          && alignment.equals(((AlignedRuleWritable) o).alignment);
    return false;
  }

  public int hashCode() {
    int result = 163;
    result = 37 * result + rule.hashCode();
    result = 37 * result + alignment.hashCode();
    return result;
  }

  public String toString() {
    StringBuilder sb = new StringBuilder();
    sb.append(rule.toString());
    sb.append(FormatUtils.DELIM);
    sb.append(alignment.toString());
    return sb.toString();
  }

  public int compareTo(AlignedRuleWritable that) {
    int cmp = this.rule.compareTo(that.rule);
    if (cmp != 0) return cmp;
    return this.alignment.compareTo(that.alignment);
  }

  public static class RuleYieldPartitioner extends Partitioner<AlignedRuleWritable, Writable> {
    public int getPartition(AlignedRuleWritable key, Writable value, int num_partitions) {
      return (key.rule.hashCode() & Integer.MAX_VALUE) % num_partitions;
    }
  }

  static {
    WritableComparator.define(AlignedRuleWritable.class, new RuleYieldComparator());
  }

  public static class RuleYieldComparator extends WritableComparator {

    private final static WritableComparator RULE = new RuleWritable.YieldComparator();
    private final static WritableComparator ALIGNMENT = new AlignmentWritable.AlignmentComparator();

    public RuleYieldComparator() {
      super(AlignedRuleWritable.class);
    }

    public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
      try {
        int cmp = RULE.compare(b1, s1, l1, b2, s2, l2);
        if (cmp != 0) return cmp;
        int a1 = RuleWritable.size(b1, s1, l1);
        int a2 = RuleWritable.size(b2, s2, l2);
        return ALIGNMENT.compare(b1, s1 + a1, l1 - a1, b2, s2 + a2, l2 - a2);
      } catch (IOException e) {
        throw new IllegalArgumentException(e);
      }
    }
  }

}
