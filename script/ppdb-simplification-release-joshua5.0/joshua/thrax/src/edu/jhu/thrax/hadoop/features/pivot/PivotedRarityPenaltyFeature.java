package edu.jhu.thrax.hadoop.features.pivot;

import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Writable;

import edu.jhu.thrax.hadoop.datatypes.FeatureMap;
import edu.jhu.thrax.hadoop.features.annotation.RarityPenaltyFeature;
import edu.jhu.thrax.util.Vocabulary;

public class PivotedRarityPenaltyFeature implements PivotedFeature {

  public static final String NAME = RarityPenaltyFeature.NAME;
  public static final String LABEL = RarityPenaltyFeature.LABEL;

  private static final FloatWritable ZERO = new FloatWritable(0.0f);

  private static final float RENORMALIZE = (float) Math.exp(-1);

  private float aggregated_rp;

  public String getName() {
    return NAME;
  }

  public String getLabel() {
    return LABEL;
  }

  public Set<String> getPrerequisites() {
    Set<String> prereqs = new HashSet<String>();
    prereqs.add(RarityPenaltyFeature.NAME);
    return prereqs;
  }

  public FloatWritable pivot(FeatureMap a, FeatureMap b) {
    float a_rp = ((FloatWritable) a.get(RarityPenaltyFeature.LABEL)).get();
    float b_rp = ((FloatWritable) b.get(RarityPenaltyFeature.LABEL)).get();
    return new FloatWritable(Math.max(a_rp, b_rp));
  }

  public void unaryGlueRuleScore(int nt, Map<Integer, Writable> map) {
    map.put(Vocabulary.id(LABEL), ZERO);
  }

  public void binaryGlueRuleScore(int nt, Map<Integer, Writable> map) {
    map.put(Vocabulary.id(LABEL), ZERO);
  }

  public void initializeAggregation() {
    aggregated_rp = -1;
  }

  public void aggregate(FeatureMap a) {
    float rp = ((FloatWritable) a.get(LABEL)).get();
    if (aggregated_rp == -1) {
      aggregated_rp = rp;
    } else {
      // Rarity is exp(1 - count). To compute rarity over a sum of counts:
      // rarity_{1+2} = exp(1 - (count_1 + count_2)) = exp(1 - count_1) * exp(-count_2) =
      // = exp(1 - count_1) * exp(1 - count_2) * exp(-1) = rarity_1 * rarity_2 * exp(-1)
      aggregated_rp *= rp * RENORMALIZE;
    }
  }

  public FloatWritable finalizeAggregation() {
    return new FloatWritable(aggregated_rp);
  }

  @Override
  public Set<String> getLowerBoundLabels() {
    Set<String> lower_bound_labels = new HashSet<String>();
    lower_bound_labels.add(RarityPenaltyFeature.LABEL);
    return lower_bound_labels;
  }

  @Override
  public Set<String> getUpperBoundLabels() {
    return null;
  }
}
