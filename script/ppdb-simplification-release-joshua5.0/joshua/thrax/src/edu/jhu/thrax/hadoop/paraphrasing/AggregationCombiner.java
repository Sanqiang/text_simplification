package edu.jhu.thrax.hadoop.paraphrasing;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Reducer;

import edu.jhu.thrax.hadoop.datatypes.FeatureMap;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.hadoop.features.annotation.AnnotationFeature;
import edu.jhu.thrax.hadoop.features.annotation.AnnotationFeatureFactory;
import edu.jhu.thrax.hadoop.features.pivot.PivotedAnnotationFeature;
import edu.jhu.thrax.hadoop.features.pivot.PivotedFeature;
import edu.jhu.thrax.hadoop.features.pivot.PivotedFeatureFactory;
import edu.jhu.thrax.util.BackwardsCompatibility;
import edu.jhu.thrax.util.FormatUtils;
import edu.jhu.thrax.util.Vocabulary;

public class AggregationCombiner
    extends Reducer<RuleWritable, FeatureMap, RuleWritable, FeatureMap> {

  private List<PivotedFeature> pivotedFeatures;

  protected void setup(Context context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    String vocabulary_path = conf.getRaw("thrax.work-dir") + "vocabulary/part-*";
    Vocabulary.initialize(conf, vocabulary_path);

    pivotedFeatures = new ArrayList<PivotedFeature>();
    List<AnnotationFeature> annotationFeatures = new ArrayList<AnnotationFeature>();

    String features = BackwardsCompatibility.equivalent(conf.get("thrax.features", ""));
    for (String f_name : FormatUtils.P_COMMA_OR_SPACE.split(features)) {
      PivotedFeature pf = PivotedFeatureFactory.get(f_name);
      if (pf != null) {
        pivotedFeatures.add(pf);
      } else {
        AnnotationFeature af = AnnotationFeatureFactory.get(f_name);
        if (af != null) {
          annotationFeatures.add(af);
        }
      }
    }
    if (!annotationFeatures.isEmpty()) pivotedFeatures.add(new PivotedAnnotationFeature());
  }

  protected void reduce(RuleWritable key, Iterable<FeatureMap> values, Context context)
      throws IOException, InterruptedException {
    FeatureMap merged = new FeatureMap();

    for (PivotedFeature feature : pivotedFeatures)
      feature.initializeAggregation();
    for (FeatureMap feature_map : values) {
      for (PivotedFeature feature : pivotedFeatures) {
        try {
          feature.aggregate(feature_map);
        } catch (Exception e) {
          throw new RuntimeException(key.toString() + " on " + feature.getName() + ": "
              + e.getMessage());
        }
      }
    }
    for (PivotedFeature feature : pivotedFeatures)
      merged.put(feature.getLabel(), feature.finalizeAggregation());
    context.write(key, merged);
  }
}
