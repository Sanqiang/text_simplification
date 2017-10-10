package edu.jhu.thrax.hadoop.features.pivot;

import java.util.HashSet;
import java.util.Set;

import org.apache.hadoop.io.FloatWritable;

import edu.jhu.thrax.hadoop.datatypes.FeatureMap;
import edu.jhu.thrax.hadoop.features.mapred.SourcePhraseGivenTargetandLHSFeature;
import edu.jhu.thrax.hadoop.features.mapred.TargetPhraseGivenSourceandLHSFeature;

public class PivotedTargetPhraseGivenSourceAndLHSFeature extends PivotedNegLogProbFeature {

  public static final String NAME = TargetPhraseGivenSourceandLHSFeature.NAME;
  public static final String LABEL = TargetPhraseGivenSourceandLHSFeature.LABEL;

  public String getName() {
    return NAME;
  }

  public String getLabel() {
    return LABEL;
  }

  public Set<String> getPrerequisites() {
    Set<String> prereqs = new HashSet<String>();
    prereqs.add(TargetPhraseGivenSourceandLHSFeature.NAME);
    prereqs.add(SourcePhraseGivenTargetandLHSFeature.NAME);
    return prereqs;
  }

  public FloatWritable pivot(FeatureMap src, FeatureMap tgt) {
    float fge = ((FloatWritable) tgt.get(TargetPhraseGivenSourceandLHSFeature.LABEL)).get();
    float egf = ((FloatWritable) src.get(SourcePhraseGivenTargetandLHSFeature.LABEL)).get();

    return new FloatWritable(egf + fge);
  }

  @Override
  public Set<String> getLowerBoundLabels() {
    Set<String> lower_bound_labels = new HashSet<String>();
    lower_bound_labels.add(TargetPhraseGivenSourceandLHSFeature.LABEL);
    lower_bound_labels.add(SourcePhraseGivenTargetandLHSFeature.LABEL);
    return lower_bound_labels;
  }

  @Override
  public Set<String> getUpperBoundLabels() {
    return null;
  }
}
