package edu.jhu.thrax.hadoop.paraphrasing;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Reducer;

import edu.jhu.thrax.hadoop.datatypes.FeatureMap;
import edu.jhu.thrax.hadoop.datatypes.FeaturePair;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.util.Vocabulary;

public class FeatureCollectionReducer
    extends Reducer<RuleWritable, FeaturePair, RuleWritable, FeatureMap> {

  protected void setup(Context context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    String vocabulary_path = conf.getRaw("thrax.work-dir") + "vocabulary/part-*";
    Vocabulary.initialize(conf, vocabulary_path);
  }

  protected void reduce(RuleWritable key, Iterable<FeaturePair> values, Context context)
      throws IOException, InterruptedException {
    FeatureMap features = new FeatureMap();
    for (FeaturePair fp : values)
      features.put(fp.key, fp.val.get());
    context.write(key, features);
  }
}
