package edu.jhu.thrax.hadoop.paraphrasing;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Mapper;

import edu.jhu.thrax.hadoop.datatypes.FeatureMap;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.util.Vocabulary;

public class AggregationMapper extends Mapper<RuleWritable, FeatureMap, RuleWritable, FeatureMap> {

  protected void setup(Context context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    Vocabulary.initialize(conf);
  }

  protected void map(RuleWritable key, FeatureMap value, Context context) throws IOException,
      InterruptedException {
    context.write(key, value);
    context.progress();
  }
}
