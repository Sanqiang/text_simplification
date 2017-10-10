package edu.jhu.thrax.hadoop.features.pivot;

import java.util.ArrayList;
import java.util.List;

import edu.jhu.thrax.util.FormatUtils;

public class PivotedFeatureFactory {

  public static PivotedFeature get(String name) {
    if (name.equals(PivotedTargetPhraseGivenSourceFeature.NAME))
      return new PivotedTargetPhraseGivenSourceFeature();
    else if (name.equals(PivotedSourcePhraseGivenTargetFeature.NAME))
      return new PivotedSourcePhraseGivenTargetFeature();
    else if (name.equals(PivotedRarityPenaltyFeature.NAME))
      return new PivotedRarityPenaltyFeature();
    else if (name.equals(PivotedLexicalSourceGivenTargetFeature.NAME))
      return new PivotedLexicalSourceGivenTargetFeature();
    else if (name.equals(PivotedLexicalTargetGivenSourceFeature.NAME))
      return new PivotedLexicalTargetGivenSourceFeature();
    else if (name.equals(PivotedSourcePhraseGivenLHSFeature.NAME))
      return new PivotedSourcePhraseGivenLHSFeature();
    else if (name.equals(PivotedLhsGivenSourcePhraseFeature.NAME))
      return new PivotedLhsGivenSourcePhraseFeature();
    else if (name.equals(PivotedSourcePhraseGivenTargetAndLHSFeature.NAME))
      return new PivotedSourcePhraseGivenTargetAndLHSFeature();
    else if (name.equals(PivotedTargetPhraseGivenLHSFeature.NAME))
      return new PivotedTargetPhraseGivenLHSFeature();
    else if (name.equals(PivotedLhsGivenTargetPhraseFeature.NAME))
      return new PivotedLhsGivenTargetPhraseFeature();
    else if (name.equals(PivotedTargetPhraseGivenSourceAndLHSFeature.NAME))
      return new PivotedTargetPhraseGivenSourceAndLHSFeature();

    return null;
  }

  public static List<PivotedFeature> getAll(String names) {
    String[] feature_names = FormatUtils.P_COMMA_OR_SPACE.split(names);
    List<PivotedFeature> features = new ArrayList<PivotedFeature>();

    for (String feature_name : feature_names) {
      PivotedFeature feature = get(feature_name);
      if (feature != null) features.add(feature);
    }
    return features;
  }
}
