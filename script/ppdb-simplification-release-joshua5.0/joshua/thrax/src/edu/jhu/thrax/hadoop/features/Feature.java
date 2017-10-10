package edu.jhu.thrax.hadoop.features;

import java.util.Map;

import org.apache.hadoop.io.Writable;

public interface Feature {
  
  public String getName();
  
  public String getLabel();
  
  public void unaryGlueRuleScore(int nt, Map<Integer, Writable> map);

  public void binaryGlueRuleScore(int nt, Map<Integer, Writable> map);

}
