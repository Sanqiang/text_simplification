package edu.jhu.thrax.hadoop.jobs;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.reduce.IntSumReducer;

import edu.jhu.thrax.hadoop.features.WordLexicalProbabilityCalculator;

public abstract class WordLexprobJob implements ThraxJob {
  public static final String SOURCE_GIVEN_TARGET = "thrax.__wordlexprob_sgt";
  private boolean isSourceGivenTarget;

  public WordLexprobJob(boolean isSrcGivenTgt) {
    isSourceGivenTarget = isSrcGivenTgt;
  }

  public Set<Class<? extends ThraxJob>> getPrerequisites() {
    Set<Class<? extends ThraxJob>> result = new HashSet<Class<? extends ThraxJob>>();
    result.add(VocabularyJob.class);
    return result;
  }

  public Job getJob(Configuration conf) throws IOException {
    Configuration theConf = new Configuration(conf);
    theConf.setBoolean(SOURCE_GIVEN_TARGET, isSourceGivenTarget);
    Job job = new Job(theConf, getName());
    job.setJarByClass(WordLexicalProbabilityCalculator.class);
    job.setMapperClass(WordLexicalProbabilityCalculator.Map.class);
    job.setCombinerClass(IntSumReducer.class);
    
    job.setPartitionerClass(WordLexicalProbabilityCalculator.Partition.class);
    job.setReducerClass(WordLexicalProbabilityCalculator.Reduce.class);

    job.setMapOutputKeyClass(LongWritable.class);
    job.setMapOutputValueClass(IntWritable.class);

    job.setOutputKeyClass(LongWritable.class);
    job.setOutputValueClass(FloatWritable.class);

    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    FileInputFormat.setInputPaths(job, new Path(conf.get("thrax.input-file")));
    int maxSplitSize = conf.getInt("thrax.max-split-size", 0);
    if (maxSplitSize != 0) {
      FileInputFormat.setMaxInputSplitSize(job, maxSplitSize);
    }
    return job;
  }
}
