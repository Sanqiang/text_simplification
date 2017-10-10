package edu.jhu.thrax.hadoop.datatypes;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableUtils;

import edu.jhu.thrax.hadoop.features.annotation.AnnotationPassthroughFeature;
import edu.jhu.thrax.util.Vocabulary;

public class FeatureMap implements Writable {

  private Map<Integer, Writable> map;

  public FeatureMap() {
    map = new HashMap<Integer, Writable>();
  }

  public FeatureMap(FeatureMap fm) {
    this();
    for (int key : fm.map.keySet())
      this.map.put(key, fm.map.get(key));
  }

  public Writable get(int key) {
    return map.get(key);
  }

  public Writable get(String key) {
    return map.get(Vocabulary.id(key));
  }

  public void put(int key, Writable val) {
    map.put(key, val);
  }
  
  public void put(String key, Writable val) {
    map.put(Vocabulary.id(key), val);
  }
  
  public boolean containsKey(int key) {
    return map.containsKey(key);
  }
  
  public Set<Integer> keySet() {
    return map.keySet();
  }
  
  @Override
  public void readFields(DataInput in) throws IOException {
    map.clear();
    int size = WritableUtils.readVInt(in);
    for (int i = 0; i < size; ++i) {
      int key = 0;
      Writable val = null;
      key = WritableUtils.readVInt(in);
      if (key == Vocabulary.id(AnnotationPassthroughFeature.LABEL)) {
        val = new Annotation();
        val.readFields(in);
      } else {
        val = new FloatWritable();
        val.readFields(in);
      }
      map.put(key, val);
    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    WritableUtils.writeVInt(out, map.size());
    for (int key : map.keySet()) {
      WritableUtils.writeVInt(out, key);
      if (key == Vocabulary.id(AnnotationPassthroughFeature.LABEL)) {
        ((Annotation) this.get(key)).write(out);
      } else {
        ((FloatWritable) this.get(key)).write(out);
      }
    }
  }
}
