package edu.jhu.thrax.hadoop.extraction;

import java.io.IOException;

import org.apache.hadoop.mapreduce.Reducer;

import edu.jhu.thrax.hadoop.datatypes.AlignedRuleWritable;
import edu.jhu.thrax.hadoop.datatypes.Annotation;

public class ExtractionCombiner extends Reducer<AlignedRuleWritable, Annotation, AlignedRuleWritable, Annotation> {

  protected void reduce(AlignedRuleWritable key, Iterable<Annotation> values, Context context)
      throws IOException, InterruptedException {
    context.progress();
    Annotation merged = new Annotation();
    for (Annotation a : values) merged.merge(a);
    context.write(key, merged);
  }
}
