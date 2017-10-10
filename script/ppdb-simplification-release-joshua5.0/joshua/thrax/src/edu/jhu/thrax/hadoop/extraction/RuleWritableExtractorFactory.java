package edu.jhu.thrax.hadoop.extraction;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import edu.jhu.thrax.hadoop.datatypes.AlignedRuleWritable;
import edu.jhu.thrax.hadoop.datatypes.Annotation;

public class RuleWritableExtractorFactory {
  public static RuleWritableExtractor create(
      Mapper<LongWritable, Text, AlignedRuleWritable, Annotation>.Context context) {
    return new HierarchicalRuleWritableExtractor(context);
  }
}
