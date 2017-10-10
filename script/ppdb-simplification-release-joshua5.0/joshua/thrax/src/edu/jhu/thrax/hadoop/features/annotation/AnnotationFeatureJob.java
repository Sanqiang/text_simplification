package edu.jhu.thrax.hadoop.features.annotation;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

import edu.jhu.thrax.hadoop.datatypes.Annotation;
import edu.jhu.thrax.hadoop.datatypes.FeaturePair;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.hadoop.jobs.ExtractionJob;
import edu.jhu.thrax.hadoop.jobs.ThraxJob;

public class AnnotationFeatureJob implements ThraxJob {

  public AnnotationFeatureJob() {}

  protected static HashSet<Class<? extends ThraxJob>> prereqs =
      new HashSet<Class<? extends ThraxJob>>();

  public Set<Class<? extends ThraxJob>> getPrerequisites() {
    prereqs.add(ExtractionJob.class);
    return prereqs;
  }

  public static void addPrerequisites(Iterable<Class<? extends ThraxJob>> cs) {
    if (cs != null)
      for (Class<? extends ThraxJob> c : cs)
        prereqs.add(c);
  }

  public static void addPrerequisite(Class<? extends ThraxJob> c) {
    prereqs.add(c);
  }

  public String getOutputSuffix() {
    return getName();
  }

  public Job getJob(Configuration conf) throws IOException {
    String name = getName();
    Job job = new Job(conf, name);
    job.setJarByClass(this.getClass());

    job.setMapperClass(Mapper.class);
    job.setPartitionerClass(RuleWritable.YieldPartitioner.class);
    job.setReducerClass(AnnotationReducer.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setMapOutputKeyClass(RuleWritable.class);
    job.setMapOutputValueClass(Annotation.class);
    job.setOutputKeyClass(RuleWritable.class);
    job.setOutputValueClass(FeaturePair.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);

    int num_reducers = conf.getInt("thrax.reducers", 4);
    job.setNumReduceTasks(num_reducers);

    FileInputFormat.setInputPaths(job, new Path(conf.get("thrax.work-dir") + "rules"));
    FileOutputFormat.setOutputPath(job, new Path(conf.get("thrax.work-dir") + "annotation"));
    return job;
  }

  @Override
  public String getName() {
    return "annotation";
  }
}
