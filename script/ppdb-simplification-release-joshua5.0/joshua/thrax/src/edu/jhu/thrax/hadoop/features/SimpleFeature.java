package edu.jhu.thrax.hadoop.features;

import org.apache.hadoop.io.Writable;

import edu.jhu.thrax.hadoop.datatypes.RuleWritable;

public interface SimpleFeature extends Feature {

  public Writable score(RuleWritable r);
  
}
