package edu.jhu.thrax.hadoop.features.annotation;

import java.util.ArrayList;
import java.util.List;

import edu.jhu.thrax.util.FormatUtils;

public class AnnotationFeatureFactory {

  public static AnnotationFeature get(String name) {
    if (name.equals(UnalignedSourceCounterFeature.NAME))
      return new UnalignedSourceCounterFeature();
    else if (name.equals(UnalignedTargetCounterFeature.NAME))
      return new UnalignedTargetCounterFeature();
    else if (name.equals(RarityPenaltyFeature.NAME))
      return new RarityPenaltyFeature();
    else if (name.equals(CountFeature.NAME))
      return new CountFeature();
    else if (name.equals(LogCountFeature.NAME))
      return new LogCountFeature();
    else if (name.equals(SourceGivenTargetLexicalProbabilityFeature.NAME))
      return new SourceGivenTargetLexicalProbabilityFeature();
    else if (name.equals(TargetGivenSourceLexicalProbabilityFeature.NAME))
      return new TargetGivenSourceLexicalProbabilityFeature();
    else if (name.equals(AlignmentFeature.NAME)) return new AlignmentFeature();

    return null;
  }

  public static List<AnnotationFeature> getAll(String names) {
    String[] feature_names = FormatUtils.P_COMMA_OR_SPACE.split(names);
    List<AnnotationFeature> features = new ArrayList<AnnotationFeature>();

    for (String feature_name : feature_names) {
      AnnotationFeature feature = get(feature_name);
      if (feature != null) features.add(feature);
    }
    return features;
  }
}
