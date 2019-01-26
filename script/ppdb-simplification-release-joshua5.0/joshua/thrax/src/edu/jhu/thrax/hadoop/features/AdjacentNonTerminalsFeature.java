package edu.jhu.thrax.hadoop.features;

import java.util.Map;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;

import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.util.Vocabulary;

public class AdjacentNonTerminalsFeature implements SimpleFeature {

  public static final String NAME = "adjacent";
  public static final String LABEL = "Adjacent";

  private static final IntWritable ZERO = new IntWritable(0);
  private static final IntWritable ONE = new IntWritable(1);

  public Writable score(RuleWritable r) {
    for (int i = 0; i < r.source.length - 1; ++i)
      if (Vocabulary.nt(r.source[i])) {
        if (Vocabulary.nt(r.source[i + 1])) {
          return ONE;
        } else {
          i += 2;
          continue;
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
    map.put(Vocabulary.id(LABEL), ONE);
  }
}
