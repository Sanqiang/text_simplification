package edu.jhu.thrax.hadoop.features.annotation;

import java.util.Map;
import java.util.Set;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Reducer.Context;

import edu.jhu.thrax.hadoop.datatypes.Annotation;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.hadoop.jobs.ThraxJob;
import edu.jhu.thrax.util.Vocabulary;

@SuppressWarnings("rawtypes")
public class RarityPenaltyFeature implements AnnotationFeature {

  public static final String NAME = "rarity";
  public static final String LABEL = "RarityPenalty";

  private static final FloatWritable ZERO = new FloatWritable(0.0f);

  public String getName() {
    return NAME;
  }

  public String getLabel() {
    return LABEL;
  }

  public void unaryGlueRuleScore(int nt, Map<Integer, Writable> map) {
    map.put(Vocabulary.id(LABEL), ZERO);
  }

  public void binaryGlueRuleScore(int nt, Map<Integer, Writable> map) {
    map.put(Vocabulary.id(LABEL), ZERO);
  }

  @Override
  public Writable score(RuleWritable r, Annotation annotation) {
    return new FloatWritable((float) Math.exp(1 - annotation.count()));
  }

  @Override
  public void init(Context context) {}

  @Override
  public Set<Class<? extends ThraxJob>> getPrerequisites() {
    return null;
  }
}
