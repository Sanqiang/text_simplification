package edu.jhu.thrax.hadoop.features.annotation;

import java.io.IOException;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Reducer.Context;

import edu.jhu.thrax.hadoop.datatypes.Annotation;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.hadoop.jobs.ThraxJob;
import edu.jhu.thrax.util.Vocabulary;

@SuppressWarnings("rawtypes")
public class UnalignedSourceCounterFeature implements AnnotationFeature {
  
  public static final String NAME = "unaligned-source";
  public static final String LABEL = "UnalignedSource";
  
  private static final IntWritable ZERO = new IntWritable(0);

  public String getName() {
    return NAME;
  }
  
  public String getLabel() {
    return LABEL;
  }

  public IntWritable score(RuleWritable r, Annotation annotation) {
    byte[] f2e = annotation.f2e().points;
    int[] src = r.source;

    int count = 0;
    int i = 0, j = 0;
    for (i = 0; i < src.length; ++i) {
      if (Vocabulary.nt(src[i])) continue;
      if (j >= f2e.length || i != f2e[j]) count++;
      while (j < f2e.length && f2e[j] <= i)
        j += 2;
    }
    return new IntWritable(count);
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
