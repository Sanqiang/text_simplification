package edu.jhu.thrax.hadoop.jobs;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class SourceWordGivenTargetWordProbabilityJob extends WordLexprobJob {

  public SourceWordGivenTargetWordProbabilityJob() {
    super(true);
  }

  public Job getJob(Configuration conf) throws IOException {
    Job job = super.getJob(conf);
    FileOutputFormat.setOutputPath(job, new Path(conf.get("thrax.work-dir") + "lexprobs_sgt"));
    return job;
  }

  public String getName() {
    return "source-word-lexprob";
  }

  public String getOutputSuffix() {
    return "lexprobs_sgt";
  }
}
