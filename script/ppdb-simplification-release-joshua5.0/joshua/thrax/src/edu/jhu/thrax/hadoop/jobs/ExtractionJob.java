package edu.jhu.thrax.hadoop.jobs;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import edu.jhu.thrax.hadoop.datatypes.AlignedRuleWritable;
import edu.jhu.thrax.hadoop.datatypes.Annotation;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.hadoop.extraction.ExtractionCombiner;
import edu.jhu.thrax.hadoop.extraction.ExtractionMapper;
import edu.jhu.thrax.hadoop.extraction.ExtractionReducer;

public class ExtractionJob implements ThraxJob {

  public Set<Class<? extends ThraxJob>> getPrerequisites() {
    Set<Class<? extends ThraxJob>> result = new HashSet<Class<? extends ThraxJob>>();
    result.add(VocabularyJob.class);
    return result;
  }

  public Job getJob(Configuration conf) throws IOException {
    Job job = new Job(conf, "extraction");
    job.setJarByClass(ExtractionMapper.class);

    job.setMapperClass(ExtractionMapper.class);
    job.setCombinerClass(ExtractionCombiner.class);
    job.setReducerClass(ExtractionReducer.class);

    job.setSortComparatorClass(AlignedRuleWritable.RuleYieldComparator.class);
    job.setPartitionerClass(AlignedRuleWritable.RuleYieldPartitioner.class);
    
    job.setMapOutputKeyClass(AlignedRuleWritable.class);
    job.setMapOutputValueClass(Annotation.class);
    job.setOutputKeyClass(RuleWritable.class);
    job.setOutputValueClass(Annotation.class);

    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    int numReducers = conf.getInt("thrax.reducers", 4);
    job.setNumReduceTasks(numReducers);

    FileInputFormat.setInputPaths(job, new Path(conf.get("thrax.input-file")));
    int maxSplitSize = conf.getInt("thrax.max-split-size", 0);
    if (maxSplitSize != 0) FileInputFormat.setMaxInputSplitSize(job, maxSplitSize);

    FileOutputFormat.setOutputPath(job, new Path(conf.get("thrax.work-dir") + "rules"));

    return job;
  }

  // TODO: unify names of jobs and their output directories

  public String getName() {
    return "extraction";
  }

  public String getOutputSuffix() {
    return "rules";
  }
}
