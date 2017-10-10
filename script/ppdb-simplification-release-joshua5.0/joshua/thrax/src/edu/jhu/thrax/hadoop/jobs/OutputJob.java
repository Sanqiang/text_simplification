package edu.jhu.thrax.hadoop.jobs;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import edu.jhu.thrax.hadoop.datatypes.FeaturePair;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.hadoop.features.annotation.AnnotationFeatureFactory;
import edu.jhu.thrax.hadoop.features.mapred.MapReduceFeature;
import edu.jhu.thrax.hadoop.features.mapred.MapReduceFeatureFactory;
import edu.jhu.thrax.hadoop.output.OutputReducer;
import edu.jhu.thrax.util.BackwardsCompatibility;
import edu.jhu.thrax.util.FormatUtils;

public class OutputJob implements ThraxJob {

  protected static HashSet<Class<? extends ThraxJob>> prereqs =
      new HashSet<Class<? extends ThraxJob>>();

  public Set<Class<? extends ThraxJob>> getPrerequisites() {
    return prereqs;
  }

  public static void addPrerequisite(Class<? extends ThraxJob> c) {
    prereqs.add(c);
  }

  public Job getJob(Configuration conf) throws IOException {
    Job job = new Job(conf, "collect");
    String workDir = conf.get("thrax.work-dir");
    job.setJarByClass(OutputReducer.class);
    job.setMapperClass(Mapper.class);
    job.setReducerClass(OutputReducer.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setMapOutputKeyClass(RuleWritable.class);
    job.setMapOutputValueClass(FeaturePair.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(NullWritable.class);

    job.setSortComparatorClass(RuleWritable.YieldComparator.class);
    job.setPartitionerClass(RuleWritable.FirstWordPartitioner.class);

    // Output is always running alone, so give it as many
    // reduce tasks as possible.
    int numReducers = conf.getInt("thrax.reducers", 4);
    job.setNumReduceTasks(numReducers);

    boolean annotation_features = false;
    String features = BackwardsCompatibility.equivalent(conf.get("thrax.features", ""));
    for (String feature : FormatUtils.P_SPACE.split(features)) {
      if (MapReduceFeatureFactory.get(feature) instanceof MapReduceFeature)
        FileInputFormat.addInputPath(job, new Path(workDir + feature));
      if (AnnotationFeatureFactory.get(feature) != null) annotation_features = true;
    }
    if (annotation_features) FileInputFormat.addInputPath(job, new Path(workDir + "annotation"));
    
    int maxSplitSize = conf.getInt("thrax.max-split-size", 0);
    if (maxSplitSize != 0) FileInputFormat.setMaxInputSplitSize(job, maxSplitSize * 20);

    if (FileInputFormat.getInputPaths(job).length == 0) {
      // TODO: This is going to crash.
      FileInputFormat.addInputPath(job, new Path(workDir + "rules"));
    } else {
      // We have at least one feature to aggregate, so we don't need
      // the rules sub-directory at all at this point.
      // We delete it to save disk space.
      final FileSystem fs = FileSystem.get(conf);
      final Path rulesPath = new Path(workDir + "rules");
      final boolean recursive = true;
      fs.delete(rulesPath, recursive);
    }

    String outputPath = conf.get("thrax.outputPath", "");
    FileOutputFormat.setOutputPath(job, new Path(outputPath));
    FileOutputFormat.setCompressOutput(job, true);
    FileOutputFormat.setOutputCompressorClass(job, GzipCodec.class);

    return job;
  }

  public String getName() {
    return "collect";
  }

  public String getOutputSuffix() {
    return null;
  }
}
