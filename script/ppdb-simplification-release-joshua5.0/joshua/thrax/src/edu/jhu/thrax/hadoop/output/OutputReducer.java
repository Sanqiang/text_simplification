package edu.jhu.thrax.hadoop.output;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Reducer;

import edu.jhu.thrax.hadoop.datatypes.FeaturePair;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.hadoop.features.SimpleFeature;
import edu.jhu.thrax.hadoop.features.SimpleFeatureFactory;
import edu.jhu.thrax.util.BackwardsCompatibility;
import edu.jhu.thrax.util.FormatUtils;
import edu.jhu.thrax.util.Vocabulary;

public class OutputReducer extends Reducer<RuleWritable, FeaturePair, Text, NullWritable> {

  private boolean label;
  private boolean sparse;

  private List<SimpleFeature> simpleFeatures;

  protected void setup(Context context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    String vocabulary_path = conf.getRaw("thrax.work-dir") + "vocabulary/part-*";
    Vocabulary.initialize(conf, vocabulary_path);
    
    label = conf.getBoolean("thrax.label-feature-scores", true);
    sparse = conf.getBoolean("thrax.sparse-feature-vectors", false);

    String features = BackwardsCompatibility.equivalent(conf.get("thrax.features", ""));
    simpleFeatures = SimpleFeatureFactory.getAll(features);
  }

  protected void reduce(RuleWritable key, Iterable<FeaturePair> values, Context context)
      throws IOException, InterruptedException {
    Map<String, Writable> features = new TreeMap<String, Writable>();
    for (FeaturePair fp : values)
      features.put(Vocabulary.word(fp.key), fp.val.get());
    for (SimpleFeature feature : simpleFeatures)
      features.put(feature.getLabel(), feature.score(key));
    context.write(FormatUtils.ruleToText(key, features, label, sparse), NullWritable.get());
  }
}
