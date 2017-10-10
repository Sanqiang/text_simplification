package edu.jhu.thrax.hadoop.features.pivot;

import java.util.Map;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Writable;

import edu.jhu.thrax.hadoop.datatypes.FeatureMap;
import edu.jhu.thrax.util.NegLogMath;
import edu.jhu.thrax.util.Vocabulary;

public abstract class PivotedNegLogProbFeature implements PivotedFeature {

  private static final FloatWritable ONE_PROB = new FloatWritable(0.0f);

  private float aggregated;

  public void initializeAggregation() {
    aggregated = 64.0f;
  }

  public void aggregate(FeatureMap features) {
    FloatWritable val = (FloatWritable) features.get(getLabel());
    aggregated = NegLogMath.logAdd(aggregated, val.get());
  }

  public FloatWritable finalizeAggregation() {
    return new FloatWritable(aggregated);
  }

  public void unaryGlueRuleScore(int nt, Map<Integer, Writable> map) {
    map.put(Vocabulary.id(getLabel()), ONE_PROB);
  }

  public void binaryGlueRuleScore(int nt, Map<Integer, Writable> map) {
    map.put(Vocabulary.id(getLabel()), ONE_PROB);
  }
}
