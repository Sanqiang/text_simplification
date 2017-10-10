package edu.jhu.thrax.hadoop.features.annotation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Reducer;

import edu.jhu.thrax.hadoop.datatypes.Annotation;
import edu.jhu.thrax.hadoop.datatypes.FeaturePair;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.util.BackwardsCompatibility;
import edu.jhu.thrax.util.Vocabulary;

public class AnnotationReducer extends Reducer<RuleWritable, Annotation, RuleWritable, FeaturePair> {

  private List<AnnotationFeature> annotationFeatures;

  public AnnotationReducer() {}

  protected void setup(Context context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    String vocabulary_path = conf.getRaw("thrax.work-dir") + "vocabulary/part-*";
    Vocabulary.initialize(conf, vocabulary_path);

    String features = BackwardsCompatibility.equivalent(conf.get("thrax.features", ""));

    // Paraphrasing only needs the annotation to be passed through.
    String type = conf.get("thrax.type", "translation");
    if ("paraphrasing".equals(type)) {
      annotationFeatures = new ArrayList<AnnotationFeature>();
      annotationFeatures.add(new AnnotationPassthroughFeature());
    } else {
      annotationFeatures = AnnotationFeatureFactory.getAll(features);
    }

    for (AnnotationFeature af : annotationFeatures)
      af.init(context);
  }

  protected void reduce(RuleWritable key, Iterable<Annotation> values, Context context)
      throws IOException, InterruptedException {
    for (Annotation annotation : values) {
      for (AnnotationFeature f : annotationFeatures) {
        context.write(key, new FeaturePair(Vocabulary.id(f.getLabel()), f.score(key, annotation)));
      }
    }
  }
}
