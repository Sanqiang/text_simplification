package edu.jhu.thrax.hadoop.features.mapred;

import java.util.ArrayList;
import java.util.List;

import edu.jhu.thrax.util.FormatUtils;

public class MapReduceFeatureFactory {

  public static MapReduceFeature get(String name) {
    if (name.equals(SourcePhraseGivenTargetFeature.NAME))
      return new SourcePhraseGivenTargetFeature();
    else if (name.equals(TargetPhraseGivenSourceFeature.NAME))
      return new TargetPhraseGivenSourceFeature();
    else if (name.equals(SourcePhraseGivenLHSFeature.NAME))
      return new SourcePhraseGivenLHSFeature();
    else if (name.equals(LhsGivenSourcePhraseFeature.NAME))
      return new LhsGivenSourcePhraseFeature();
    else if (name.equals(SourcePhraseGivenTargetandLHSFeature.NAME))
      return new SourcePhraseGivenTargetandLHSFeature();
    else if (name.equals(TargetPhraseGivenSourceandLHSFeature.NAME))
      return new TargetPhraseGivenSourceandLHSFeature();
    else if (name.equals(TargetPhraseGivenLHSFeature.NAME))
      return new TargetPhraseGivenLHSFeature();
    else if (name.equals(LhsGivenTargetPhraseFeature.NAME))
      return new LhsGivenTargetPhraseFeature();

    return null;
  }

  public static List<MapReduceFeature> getAll(String names) {
    String[] feature_names = FormatUtils.P_COMMA_OR_SPACE.split(names);
    List<MapReduceFeature> features = new ArrayList<MapReduceFeature>();

    for (String feature_name : feature_names) {
      MapReduceFeature feature = get(feature_name);
      if (feature != null) features.add(feature);
    }
    return features;
  }
}
