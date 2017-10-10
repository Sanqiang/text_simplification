package edu.jhu.thrax.hadoop.features.pivot;

import java.util.HashSet;
import java.util.Set;

import org.apache.hadoop.io.FloatWritable;

import edu.jhu.thrax.hadoop.datatypes.FeatureMap;
import edu.jhu.thrax.hadoop.features.mapred.TargetPhraseGivenLHSFeature;

public class PivotedTargetPhraseGivenLHSFeature extends NonAggregatingPivotedFeature {

  public static final String NAME = TargetPhraseGivenLHSFeature.NAME;
  public static final String LABEL = TargetPhraseGivenLHSFeature.LABEL;

  public String getName() {
    return NAME;
  }

  public String getLabel() {
    return LABEL;
  }

  public Set<String> getPrerequisites() {
    Set<String> prereqs = new HashSet<String>();
    prereqs.add(TargetPhraseGivenLHSFeature.NAME);
    return prereqs;
  }

  public FloatWritable pivot(FeatureMap src, FeatureMap tgt) {
    return new FloatWritable(((FloatWritable) tgt.get(TargetPhraseGivenLHSFeature.LABEL)).get());
  }

  @Override
  public Set<String> getLowerBoundLabels() {
    Set<String> lower_bound_labels = new HashSet<String>();
    lower_bound_labels.add(TargetPhraseGivenLHSFeature.LABEL);
    return lower_bound_labels;
  }

  @Override
  public Set<String> getUpperBoundLabels() {
    return null;
  }
}
