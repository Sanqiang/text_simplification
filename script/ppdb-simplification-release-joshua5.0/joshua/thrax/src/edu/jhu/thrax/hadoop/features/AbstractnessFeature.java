package edu.jhu.thrax.hadoop.features;

import java.util.Map;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;

import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.util.Vocabulary;

public class AbstractnessFeature implements SimpleFeature {
  
  public static final String NAME = "abstract";
  public static final String LABEL = "Abstract";
  
  private static final IntWritable ZERO = new IntWritable(0);
  private static final IntWritable ONE = new IntWritable(1);

  public Writable score(RuleWritable r) {
    for (int word : r.source) {
      if (!Vocabulary.nt(word)) {
        return ZERO;
      }
    }
    for (int word : r.target) {
      if (!Vocabulary.nt(word)) {
        return ZERO;
      }
    }
    return ONE;
  }
  
  public String getName() {
    return NAME;
  }

  public String getLabel() {
    return LABEL;
  }

  public void unaryGlueRuleScore(int nt, Map<Integer, Writable> map) {
    map.put(Vocabulary.id(LABEL), ONE);
  }

  public void binaryGlueRuleScore(int nt, Map<Integer, Writable> map) {
    map.put(Vocabulary.id(LABEL), ONE);
  }
}
