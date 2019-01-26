package edu.jhu.thrax.hadoop.features;

import java.util.ArrayList;
import java.util.List;

import edu.jhu.thrax.util.FormatUtils;

public class SimpleFeatureFactory {

  public static SimpleFeature get(String name) {
    if (name.equals(AbstractnessFeature.NAME))
      return new AbstractnessFeature();
    else if (name.equals(AdjacentNonTerminalsFeature.NAME))
      return new AdjacentNonTerminalsFeature();
    else if (name.equals(LexicalityFeature.NAME))
      return new LexicalityFeature();
    else if (name.equals(XRuleFeature.NAME))
      return new XRuleFeature();
    else if (name.equals(MonotonicFeature.NAME))
      return new MonotonicFeature();
    else if (name.equals(PhrasePenaltyFeature.NAME))
      return new PhrasePenaltyFeature();
    else if (name.equals(SourceWordCounterFeature.NAME))
      return new SourceWordCounterFeature();
    else if (name.equals(TargetWordCounterFeature.NAME))
      return new TargetWordCounterFeature();
    else if (name.equals(ConsumeSourceTerminalsFeature.NAME))
      return new ConsumeSourceTerminalsFeature();
    else if (name.equals(ProduceTargetTerminalsFeature.NAME))
      return new ProduceTargetTerminalsFeature();
    else if (name.equals(IdentityFeature.NAME))
      return new IdentityFeature();
    else if (name.equals(WordCountDifferenceFeature.NAME))
      return new WordCountDifferenceFeature();
    else if (name.equals(WordLengthDifferenceFeature.NAME))
      return new WordLengthDifferenceFeature();
    else if (name.equals(WordCompressionRatioFeature.NAME))
      return new WordCompressionRatioFeature();
    else if (name.equals(CharacterCountDifferenceFeature.NAME))
      return new CharacterCountDifferenceFeature();
    else if (name.equals(CharacterCompressionRatioFeature.NAME))
      return new CharacterCompressionRatioFeature();
    else if (name.equals(GlueRuleFeature.NAME)) return new GlueRuleFeature();

    return null;
  }

  public static List<SimpleFeature> getAll(String names) {
    String[] feature_names = FormatUtils.P_COMMA_OR_SPACE.split(names);
    List<SimpleFeature> features = new ArrayList<SimpleFeature>();

    for (String feature_name : feature_names) {
      SimpleFeature feature = get(feature_name);
      if (feature != null) features.add(feature);
    }
    return features;
  }
}
