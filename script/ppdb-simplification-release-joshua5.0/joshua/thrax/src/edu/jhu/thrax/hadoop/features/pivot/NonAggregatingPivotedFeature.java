package edu.jhu.thrax.hadoop.features.pivot;

import java.util.Map;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Writable;

import edu.jhu.thrax.hadoop.datatypes.FeatureMap;
import edu.jhu.thrax.util.Vocabulary;

public abstract class NonAggregatingPivotedFeature implements PivotedFeature {

  private static final FloatWritable ZERO = new FloatWritable(0.0f);

  private float value;

  public void initializeAggregation() {
    value = Float.MAX_VALUE;
  }

  public void aggregate(FeatureMap features) {
    FloatWritable val = (FloatWritable) features.get(getLabel());
    if (value == Float.MAX_VALUE) {
      value = val.get();
    } else {
      if (value != val.get()) {
        throw new RuntimeException("Diverging values in pseudo-aggregation: " + value + " versus "
            + val.get());
      }
    }
  }

  public FloatWritable finalizeAggregation() {
    return new FloatWritable(value);
  }

  public void unaryGlueRuleScore(int nt, Map<Integer, Writable> map) {
    map.put(Vocabulary.id(getLabel()), ZERO);
  }

  public void binaryGlueRuleScore(int nt, Map<Integer, Writable> map) {
    map.put(Vocabulary.id(getLabel()), ZERO);
  }
}
