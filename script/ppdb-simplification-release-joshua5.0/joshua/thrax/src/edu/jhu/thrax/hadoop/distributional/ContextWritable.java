package edu.jhu.thrax.hadoop.distributional;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.BooleanWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;

import edu.jhu.jerboa.sim.SLSH;
import edu.jhu.jerboa.sim.Signature;
import edu.jhu.thrax.hadoop.datatypes.PrimitiveUtils;

/**
 * A union-like writable that contains a set of context features.
 * 
 * @author Juri Ganitkevitch
 * 
 */
public class ContextWritable implements Writable {
  public IntWritable strength;
  public BooleanWritable compacted;
  public MapWritable map;
  public float[] sums;

  public ContextWritable() {
    this(false);
  }

  public ContextWritable(boolean compacted) {
    this.strength = new IntWritable(0);
    this.compacted = new BooleanWritable(compacted);
    if (compacted) {
      // TODO: stupid initialization, double-check.
      this.map = null;
      this.sums = null;
    } else {
      this.map = new MapWritable();
      this.sums = null;
    }
  }

  public ContextWritable(int strength, MapWritable map) {
    this.strength = new IntWritable(strength);
    this.compacted = new BooleanWritable(false);
    this.map = new MapWritable(map);
    this.sums = null;
  }

  public ContextWritable(int strength, float[] sums) {
    this.strength = new IntWritable(strength);
    this.compacted = new BooleanWritable(true);
    this.map = null;
    this.sums = sums;
  }

  public void merge(ContextWritable that, SLSH slsh) {
    if (this.compacted.get()) {
      if (!that.compacted.get()) that.compact(slsh);
      this.mergeSums(that, slsh);
    } else {
      if (that.compacted.get()) {
       this.compact(slsh);
       this.mergeSums(that, slsh);
      } else {
        for (Writable feature_text : that.map.keySet()) {
          int feature_value = ((IntWritable) that.map.get(feature_text)).get();
          IntWritable current_value = (IntWritable) this.map.get(feature_text);
          if (current_value != null)
            this.map.put(feature_text, new IntWritable(current_value.get() + feature_value));
          else
            this.map.put(feature_text, new IntWritable(feature_value));
        }
      }
    }
    this.strength = new IntWritable(this.strength.get() + that.strength.get());
  }
  
  private void mergeSums(ContextWritable that, SLSH slsh) {
    if (!that.compacted.get()) {
      throw new RuntimeException("Trying to merge sums on un-compacted ContextWritable.");
    }
    Signature this_signature = new Signature();
    Signature that_signature = new Signature();
    // TODO: probably needs deep copy.
    this_signature.sums = sums;
    that_signature.sums = sums;
    slsh.updateSignature(this_signature, that_signature);
  }

  public void compact(SLSH slsh) {
    Signature signature = new Signature();
    slsh.initializeSignature(signature);
    for (Writable feature_name : map.keySet()) {
      slsh.updateSignature(signature, ((Text) feature_name).toString(),
          ((IntWritable) map.get(feature_name)).get(), 1);
    }
    compacted.set(true);
    map = null;
    sums = signature.sums;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    strength.readFields(in);
    compacted.readFields(in);
    if (compacted.get()) {
      map = null;
      sums = PrimitiveUtils.readFloatArray(in);
    } else {
      if (map == null) map = new MapWritable();
      map.readFields(in);
      sums = null;
    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    strength.write(out);
    compacted.write(out);
    if (compacted.get()) {
      PrimitiveUtils.writeFloatArray(out, sums);
    } else {
      map.write(out);
    }
  }
}
