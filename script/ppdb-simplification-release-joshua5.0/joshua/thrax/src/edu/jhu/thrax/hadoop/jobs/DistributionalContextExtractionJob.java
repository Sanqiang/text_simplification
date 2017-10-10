package edu.jhu.thrax.hadoop.jobs;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import edu.jhu.thrax.hadoop.distributional.ContextWritable;
import edu.jhu.thrax.hadoop.distributional.DistributionalContextCombiner;
import edu.jhu.thrax.hadoop.distributional.DistributionalContextMapper;
import edu.jhu.thrax.hadoop.distributional.DistributionalContextReducer;
import edu.jhu.thrax.hadoop.distributional.SignatureWritable;

public class DistributionalContextExtractionJob implements ThraxJob {

  public Job getJob(Configuration conf) throws IOException {
    Job job = new Job(conf, "distributional");

    job.setJarByClass(DistributionalContextMapper.class);

    job.setMapperClass(DistributionalContextMapper.class);
    job.setCombinerClass(DistributionalContextCombiner.class);
    job.setReducerClass(DistributionalContextReducer.class);

    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(ContextWritable.class);

    job.setOutputKeyClass(SignatureWritable.class);
    job.setOutputValueClass(NullWritable.class);

    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    int numReducers = conf.getInt("thrax.reducers", 4);
    job.setNumReduceTasks(numReducers);

    FileInputFormat.setInputPaths(job, new Path(conf.get("thrax.input-file")));
    FileOutputFormat.setOutputPath(job, new Path(conf.get("thrax.work-dir") + "signatures"));

    int max_split_size = conf.getInt("thrax.max-split-size", 0);
    if (max_split_size != 0) FileInputFormat.setMaxInputSplitSize(job, max_split_size);

    return job;
  }

  public String getName() {
    return "distributional";
  }

  public String getOutputSuffix() {
    return null;
  }

  @Override
  public Set<Class<? extends ThraxJob>> getPrerequisites() {
    return new HashSet<Class<? extends ThraxJob>>();
  }
}
