package edu.jhu.thrax.hadoop.features.annotation;

import java.io.IOException;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Reducer.Context;

import edu.jhu.thrax.hadoop.datatypes.AlignmentWritable;
import edu.jhu.thrax.hadoop.datatypes.Annotation;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.hadoop.jobs.ThraxJob;
import edu.jhu.thrax.util.Vocabulary;

@SuppressWarnings("rawtypes")
public class AlignmentFeature implements AnnotationFeature {
  
  public static final String NAME = "alignment";
  public static final String LABEL = "Alignment";
  
  private static final IntWritable ZERO = new IntWritable(0);

  public String getName() {
    return NAME;
  }
  
  public String getLabel() {
    return LABEL;
  }

  public AlignmentWritable score(RuleWritable r, Annotation annotation) {
    return annotation.f2e();
  }

  public void unaryGlueRuleScore(int nt, Map<Integer, Writable> map) {
    map.put(Vocabulary.id(LABEL), ZERO);
  }

  public void binaryGlueRuleScore(int nt, Map<Integer, Writable> map) {
    map.put(Vocabulary.id(LABEL), ZERO);
  }

  @Override
  public void init(Context context) throws IOException, InterruptedException {}

  @Override
  public Set<Class<? extends ThraxJob>> getPrerequisites() {
    return null;
  }
}
