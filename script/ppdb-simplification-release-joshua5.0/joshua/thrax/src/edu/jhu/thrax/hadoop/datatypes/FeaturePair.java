package edu.jhu.thrax.hadoop.datatypes;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableUtils;

import edu.jhu.thrax.util.Vocabulary;

public class FeaturePair implements Writable {
  public int key;
  public FeatureValue val;

  public FeaturePair() {
    key = 0;
    val = new FeatureValue();
  }

  public FeaturePair(int k, Writable v) {
    key = k;
    val = new FeatureValue(v);
  }

  public void write(DataOutput out) throws IOException {
    WritableUtils.writeVInt(out, key);
    val.write(out);
  }

  public void readFields(DataInput in) throws IOException {
    key = WritableUtils.readVInt(in);
    val.readFields(in);
  }

  public int hashCode() {
    return key * 163 + val.hashCode();
  }

  public boolean equals(Object o) {
    if (o instanceof FeaturePair) {
      FeaturePair that = (FeaturePair) o;
      return key == that.key && val.equals(that.val);
    }
    return false;
  }

  public String toString() {
    return Vocabulary.word(key) + "=" + val.toString();
  }
}
