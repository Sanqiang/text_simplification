package edu.jhu.thrax.hadoop.jobs;

import java.io.IOException;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Job;

public interface ThraxJob {

  public Job getJob(Configuration conf) throws IOException;

  public Set<Class<? extends ThraxJob>> getPrerequisites();

  public String getName();

  public String getOutputSuffix();
}
