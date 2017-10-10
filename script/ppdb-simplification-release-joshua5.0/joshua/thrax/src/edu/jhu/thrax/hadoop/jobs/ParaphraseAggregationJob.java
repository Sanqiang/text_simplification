package edu.jhu.thrax.hadoop.jobs;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import edu.jhu.thrax.hadoop.datatypes.FeatureMap;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.hadoop.paraphrasing.AggregationCombiner;
import edu.jhu.thrax.hadoop.paraphrasing.AggregationMapper;
import edu.jhu.thrax.hadoop.paraphrasing.AggregationReducer;

public class ParaphraseAggregationJob implements ThraxJob {

  private static HashSet<Class<? extends ThraxJob>> prereqs =
      new HashSet<Class<? extends ThraxJob>>();

  public Job getJob(Configuration conf) throws IOException {
    Job job = new Job(conf, "aggregate");

    job.setJarByClass(AggregationReducer.class);

    job.setMapperClass(AggregationMapper.class);
    job.setCombinerClass(AggregationCombiner.class);
    job.setReducerClass(AggregationReducer.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setMapOutputKeyClass(RuleWritable.class);
    job.setMapOutputValueClass(FeatureMap.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(NullWritable.class);

    job.setSortComparatorClass(RuleWritable.YieldComparator.class);
    job.setPartitionerClass(RuleWritable.FirstWordPartitioner.class);

    FileInputFormat.setInputPaths(job, new Path(conf.get("thrax.work-dir") + "pivoted"));
    int maxSplitSize = conf.getInt("thrax.max-split-size", 0);
    if (maxSplitSize != 0) FileInputFormat.setMaxInputSplitSize(job, maxSplitSize * 20);

    int numReducers = conf.getInt("thrax.reducers", 4);
    job.setNumReduceTasks(numReducers);

    String outputPath = conf.get("thrax.outputPath", "");
    FileOutputFormat.setOutputPath(job, new Path(outputPath));
    
    FileOutputFormat.setCompressOutput(job, true);
    FileOutputFormat.setOutputCompressorClass(job, GzipCodec.class);

    return job;
  }

  public String getName() {
    return "aggregate";
  }

  public static void addPrerequisite(Class<? extends ThraxJob> c) {
    prereqs.add(c);
  }

  public Set<Class<? extends ThraxJob>> getPrerequisites() {
    prereqs.add(ParaphrasePivotingJob.class);
    return prereqs;
  }

  public String getOutputSuffix() {
    return null;
  }
}
