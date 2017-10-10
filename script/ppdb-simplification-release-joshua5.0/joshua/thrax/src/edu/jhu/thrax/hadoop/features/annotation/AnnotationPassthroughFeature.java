package edu.jhu.thrax.hadoop.features.annotation;

import java.io.IOException;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Reducer.Context;

import edu.jhu.thrax.hadoop.datatypes.Annotation;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.hadoop.jobs.ThraxJob;

@SuppressWarnings("rawtypes")
public class AnnotationPassthroughFeature implements AnnotationFeature {
  
  public static final String NAME = "annotation";
  public static final String LABEL = "Annotation";

  public String getName() {
    return NAME;
  }
  
  public String getLabel() {
    return LABEL;
  }

  public Annotation score(RuleWritable r, Annotation annotation) {
    return annotation;
  }

  public void unaryGlueRuleScore(int nt, Map<Integer, Writable> map) {
  }

  public void binaryGlueRuleScore(int nt, Map<Integer, Writable> map) {
  }

  @Override
  public void init(Context context) throws IOException, InterruptedException {}

  @Override
  public Set<Class<? extends ThraxJob>> getPrerequisites() {
    return null;
  }
}
