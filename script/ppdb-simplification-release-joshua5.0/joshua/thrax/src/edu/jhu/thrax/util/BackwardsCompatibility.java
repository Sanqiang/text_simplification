package edu.jhu.thrax.util;

import edu.jhu.thrax.hadoop.features.annotation.SourceGivenTargetLexicalProbabilityFeature;
import edu.jhu.thrax.hadoop.features.annotation.TargetGivenSourceLexicalProbabilityFeature;
import edu.jhu.thrax.hadoop.features.annotation.UnalignedSourceCounterFeature;
import edu.jhu.thrax.hadoop.features.annotation.UnalignedTargetCounterFeature;
import edu.jhu.thrax.hadoop.features.mapred.SourcePhraseGivenTargetFeature;
import edu.jhu.thrax.hadoop.features.mapred.TargetPhraseGivenSourceFeature;

public class BackwardsCompatibility {

  public static String equivalent(String features) {
    features = features.replace("e2fphrase", SourcePhraseGivenTargetFeature.NAME);
    features = features.replace("f2ephrase", TargetPhraseGivenSourceFeature.NAME);

    features = features.replace("lexprob_tgs", TargetGivenSourceLexicalProbabilityFeature.NAME);
    features = features.replace("lexprob_sgt", SourceGivenTargetLexicalProbabilityFeature.NAME);

    features =
        features.replace("lexprob", TargetGivenSourceLexicalProbabilityFeature.NAME + " "
            + SourceGivenTargetLexicalProbabilityFeature.NAME);

    features =
        features.replace("unaligned-count", UnalignedSourceCounterFeature.NAME + " "
            + UnalignedTargetCounterFeature.NAME);

    return features;
  }

	public static String defaultLabelPolicy(boolean allow_nonlexical_x) {
		if (allow_nonlexical_x) {
			return "always";
		} else {
			return "phrases";
		}
	}
}
