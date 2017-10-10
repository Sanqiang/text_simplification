package edu.jhu.thrax.hadoop.paraphrasing;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Reducer;

import edu.jhu.thrax.hadoop.datatypes.Annotation;
import edu.jhu.thrax.hadoop.datatypes.FeatureMap;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.hadoop.features.SimpleFeature;
import edu.jhu.thrax.hadoop.features.SimpleFeatureFactory;
import edu.jhu.thrax.hadoop.features.annotation.AnnotationFeature;
import edu.jhu.thrax.hadoop.features.annotation.AnnotationFeatureFactory;
import edu.jhu.thrax.hadoop.features.annotation.AnnotationPassthroughFeature;
import edu.jhu.thrax.hadoop.features.pivot.PivotedAnnotationFeature;
import edu.jhu.thrax.hadoop.features.pivot.PivotedFeature;
import edu.jhu.thrax.hadoop.features.pivot.PivotedFeatureFactory;
import edu.jhu.thrax.util.BackwardsCompatibility;
import edu.jhu.thrax.util.FormatUtils;
import edu.jhu.thrax.util.Vocabulary;

public class AggregationReducer extends Reducer<RuleWritable, FeatureMap, Text, NullWritable> {

  private boolean label;
  private boolean sparse;

  private List<SimpleFeature> simpleFeatures;
  private List<PivotedFeature> pivotedFeatures;
  private List<AnnotationFeature> annotationFeatures;

  protected void setup(Context context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    String vocabulary_path = conf.getRaw("thrax.work-dir") + "vocabulary/part-*";
    Vocabulary.initialize(conf, vocabulary_path);

    label = conf.getBoolean("thrax.label-feature-scores", true);
    sparse = conf.getBoolean("thrax.sparse-feature-vectors", false);

    simpleFeatures = new ArrayList<SimpleFeature>();
    pivotedFeatures = new ArrayList<PivotedFeature>();
    annotationFeatures = new ArrayList<AnnotationFeature>();

    String features = BackwardsCompatibility.equivalent(conf.get("thrax.features", ""));
    for (String f_name : FormatUtils.P_COMMA_OR_SPACE.split(features)) {
      PivotedFeature pf = PivotedFeatureFactory.get(f_name);
      if (pf != null) {
        pivotedFeatures.add(pf);
      } else {
        AnnotationFeature af = AnnotationFeatureFactory.get(f_name);
        if (af != null) {
          annotationFeatures.add(af);
        } else {
          SimpleFeature sf = SimpleFeatureFactory.get(f_name);
          if (sf != null) simpleFeatures.add(sf);
        }
      }
    }
    if (!annotationFeatures.isEmpty()) pivotedFeatures.add(new PivotedAnnotationFeature());
    for (AnnotationFeature af : annotationFeatures)
      af.init(context);
  }

  protected void reduce(RuleWritable key, Iterable<FeatureMap> values, Context context)
      throws IOException, InterruptedException {
    RuleWritable rule = new RuleWritable(key);
    TreeMap<String, Writable> features = new TreeMap<String, Writable>();

    for (PivotedFeature feature : pivotedFeatures)
      feature.initializeAggregation();
    for (FeatureMap feature_map : values) {
      for (PivotedFeature feature : pivotedFeatures) {
        try {
          feature.aggregate(feature_map);
        } catch (Exception e) {
          throw new RuntimeException(rule.toString() + " on " + feature.getName() + ": "
              + e.getMessage());
        }
      }
    }
    for (PivotedFeature feature : pivotedFeatures)
      features.put(feature.getLabel(), feature.finalizeAggregation());

    for (SimpleFeature feature : simpleFeatures)
      features.put(feature.getLabel(), feature.score(rule));

    for (AnnotationFeature feature : annotationFeatures)
      features.put(feature.getLabel(),
          feature.score(rule, (Annotation) features.get(AnnotationPassthroughFeature.LABEL)));
    features.remove(AnnotationPassthroughFeature.LABEL);

    context.write(FormatUtils.ruleToText(rule, features, label, sparse), NullWritable.get());
  }

  protected void cleanup(Context context) throws IOException, InterruptedException {}
}
