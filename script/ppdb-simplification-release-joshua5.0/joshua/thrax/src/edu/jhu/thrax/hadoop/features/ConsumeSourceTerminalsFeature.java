package edu.jhu.thrax.hadoop.features;

import java.util.Map;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;

import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.util.Vocabulary;

public class ConsumeSourceTerminalsFeature implements SimpleFeature {

  public static final String NAME = "source-terminals-without-target";
  public static final String LABEL = "SourceTerminalsButNoTarget";
  
  private static final IntWritable ZERO = new IntWritable(0);
  private static final IntWritable ONE = new IntWritable(1);

  public Writable score(RuleWritable r) {
    for (int tok : r.target) {
      if (!Vocabulary.nt(tok)) {
        return ZERO;
      }
    }
    for (int tok : r.source) {
      if (!Vocabulary.nt(tok)) {
        return ONE;
      }
    }
    return ZERO;
  }
  
  public String getName() {
    return NAME;
  }

  public String getLabel() {
    return LABEL;
  }

  public void unaryGlueRuleScore(int nt, Map<Integer, Writable> map) {
    map.put(Vocabulary.id(LABEL), ZERO);
  }

  public void binaryGlueRuleScore(int nt, Map<Integer, Writable> map) {
    map.put(Vocabulary.id(LABEL), ZERO);
  }
}
