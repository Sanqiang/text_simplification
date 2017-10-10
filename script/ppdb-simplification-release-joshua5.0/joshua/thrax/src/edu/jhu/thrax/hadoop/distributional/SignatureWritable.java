package edu.jhu.thrax.hadoop.distributional;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Partitioner;

import edu.jhu.jerboa.sim.Signature;
import edu.jhu.thrax.hadoop.datatypes.PrimitiveUtils;

public class SignatureWritable implements WritableComparable<SignatureWritable> {
  public Text key;
  public byte[] bytes;
  public IntWritable strength;

  public SignatureWritable() {
    this.key = new Text();
    this.bytes = null;
    this.strength = new IntWritable();
  }

  public SignatureWritable(Text key, Signature signature, int strength) {
    this.key = new Text(key);
    // TODO: deep copy?
    this.bytes = signature.bytes;
    this.strength = new IntWritable(strength);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    key.readFields(in);
    bytes = PrimitiveUtils.readByteArray(in);
    strength.readFields(in);
  }

  @Override
  public void write(DataOutput out) throws IOException {
    key.write(out);
    PrimitiveUtils.writeByteArray(out, bytes);
    strength.write(out);
  }

  @Override
  public int compareTo(SignatureWritable that) {
    int cmp = strength.compareTo(that.strength);
    // Flip sign for descending sort order.
    if (cmp != 0) return -cmp;
    return key.compareTo(that.key);
  }

  public static class SignaturePartitioner extends Partitioner<SignatureWritable, Writable> {
    public int getPartition(SignatureWritable signature, Writable value, int num_partitions) {
      int hash = 163;
      hash = 37 * hash + signature.key.hashCode();
      hash = 37 * hash + signature.bytes.hashCode();
      hash = 37 * hash + signature.strength.hashCode();
      return (hash & Integer.MAX_VALUE) % num_partitions;
    }
  }
}
