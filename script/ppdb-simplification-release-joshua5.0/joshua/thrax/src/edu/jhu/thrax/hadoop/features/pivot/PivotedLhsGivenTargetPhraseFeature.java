package edu.jhu.thrax.hadoop.features.pivot;

import java.util.HashSet;
import java.util.Set;

import org.apache.hadoop.io.FloatWritable;

import edu.jhu.thrax.hadoop.datatypes.FeatureMap;
import edu.jhu.thrax.hadoop.features.mapred.LhsGivenTargetPhraseFeature;

public class PivotedLhsGivenTargetPhraseFeature extends NonAggregatingPivotedFeature {

  public static final String NAME = LhsGivenTargetPhraseFeature.NAME;
  public static final String LABEL = LhsGivenTargetPhraseFeature.LABEL;

  public String getName() {
    return NAME;
  }

  public String getLabel() {
    return LABEL;
  }

  public Set<String> getPrerequisites() {
    Set<String> prereqs = new HashSet<String>();
    prereqs.add(LhsGivenTargetPhraseFeature.NAME);
    return prereqs;
  }

  public FloatWritable pivot(FeatureMap src, FeatureMap tgt) {
    return new FloatWritable(((FloatWritable) tgt.get(LhsGivenTargetPhraseFeature.LABEL)).get());
  }

  @Override
  public Set<String> getLowerBoundLabels() {
    Set<String> lower_bound_labels = new HashSet<String>();
    lower_bound_labels.add(LhsGivenTargetPhraseFeature.LABEL);
    return lower_bound_labels;
  }

  @Override
  public Set<String> getUpperBoundLabels() {
    return null;
  }
}
