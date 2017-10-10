package edu.jhu.thrax.hadoop.features.annotation;

import java.io.IOException;
import java.util.Set;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Reducer.Context;

import edu.jhu.thrax.hadoop.datatypes.Annotation;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.hadoop.features.Feature;
import edu.jhu.thrax.hadoop.jobs.ThraxJob;

public interface AnnotationFeature extends Feature {

  @SuppressWarnings("rawtypes")
  public void init(Context context) throws IOException, InterruptedException;

  public Writable score(RuleWritable r, Annotation annotation);

  // TODO: move this into its own interface, have AF extend it.
  public Set<Class<? extends ThraxJob>> getPrerequisites();
}
