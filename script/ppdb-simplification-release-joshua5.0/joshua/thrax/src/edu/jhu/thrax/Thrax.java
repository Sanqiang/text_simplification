package edu.jhu.thrax;

import java.util.Date;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import edu.jhu.thrax.hadoop.features.annotation.AnnotationFeature;
import edu.jhu.thrax.hadoop.features.annotation.AnnotationFeatureFactory;
import edu.jhu.thrax.hadoop.features.annotation.AnnotationFeatureJob;
import edu.jhu.thrax.hadoop.features.mapred.MapReduceFeature;
import edu.jhu.thrax.hadoop.features.mapred.MapReduceFeatureFactory;
import edu.jhu.thrax.hadoop.features.pivot.PivotedFeature;
import edu.jhu.thrax.hadoop.features.pivot.PivotedFeatureFactory;
import edu.jhu.thrax.hadoop.jobs.DistributionalContextExtractionJob;
import edu.jhu.thrax.hadoop.jobs.DistributionalContextSortingJob;
import edu.jhu.thrax.hadoop.jobs.ExtractionJob;
import edu.jhu.thrax.hadoop.jobs.FeatureCollectionJob;
import edu.jhu.thrax.hadoop.jobs.JobState;
import edu.jhu.thrax.hadoop.jobs.OutputJob;
import edu.jhu.thrax.hadoop.jobs.ParaphraseAggregationJob;
import edu.jhu.thrax.hadoop.jobs.ParaphrasePivotingJob;
import edu.jhu.thrax.hadoop.jobs.Scheduler;
import edu.jhu.thrax.hadoop.jobs.SchedulerException;
import edu.jhu.thrax.hadoop.jobs.ThraxJob;
import edu.jhu.thrax.hadoop.jobs.VocabularyJob;
import edu.jhu.thrax.util.BackwardsCompatibility;
import edu.jhu.thrax.util.ConfFileParser;

public class Thrax extends Configured implements Tool {
  private Scheduler scheduler;
  private Configuration conf;

  public synchronized int run(String[] argv) throws Exception {
    if (argv.length < 1) {
      System.err.println("usage: Thrax <conf file> [output path]");
      return 1;
    }
    // do some setup of configuration
    conf = getConf();
    Map<String, String> options = ConfFileParser.parse(argv[0]);
    for (String opt : options.keySet())
      conf.set("thrax." + opt, options.get(opt));
    String date = (new Date()).toString().replaceAll("\\s+", "_").replaceAll(":", "_");

    String workDir = "thrax_run_" + date + Path.SEPARATOR;

    if (argv.length > 1) {
      workDir = argv[1];
      if (!workDir.endsWith(Path.SEPARATOR)) workDir += Path.SEPARATOR;
    }

    conf.set("thrax.work-dir", workDir);
    conf.set("thrax.outputPath", workDir + "final");

    if (options.containsKey("timeout")) {
      conf.setInt("mapreduce.task.timeout", Integer.parseInt(options.get("timeout")));
      conf.setInt("mapred.task.timeout", Integer.parseInt(options.get("timeout")));
    }

    scheduleJobs();

    do {
      for (Class<? extends ThraxJob> c : scheduler.getClassesByState(JobState.READY)) {
        scheduler.setState(c, JobState.RUNNING);
        (new Thread(new ThraxJobWorker(this, c, conf))).start();
      }
      wait();
    } while (scheduler.notFinished());
    System.err.print(scheduler);
    if (scheduler.getClassesByState(JobState.SUCCESS).size() == scheduler.numJobs()) {
      System.err.println("Work directory was " + workDir);
      System.err.println("To retrieve grammar:");
      System.err.println("hadoop fs -getmerge " + conf.get("thrax.outputPath", "")
          + " <destination>");
    }
    return 0;
  }

  // Schedule all the jobs required for grammar extraction. We
  // currently distinguish three modes: translation grammar extraction,
  // paraphrase grammar extraction, and collection of distributional signatures.
  private synchronized void scheduleJobs() throws SchedulerException {
    scheduler = new Scheduler(conf);

    String type = conf.get("thrax.type", "translation");
    String features = BackwardsCompatibility.equivalent(conf.get("thrax.features", ""));

    System.err.println("Running in mode: " + type);

    scheduler.schedule(VocabularyJob.class);

    // Translation grammar mode.
    if ("translation".equals(type)) {
      // Schedule rule extraction job.
      scheduler.schedule(ExtractionJob.class);
      // Create feature map-reduces.
      for (MapReduceFeature f : MapReduceFeatureFactory.getAll(features)) {
        scheduler.schedule(f.getClass());
        OutputJob.addPrerequisite(f.getClass());
      }
      // Set up annotation-level feature & prerequisites.
      List<AnnotationFeature> annotation_features = AnnotationFeatureFactory.getAll(features);
      for (AnnotationFeature f : annotation_features)
        AnnotationFeatureJob.addPrerequisites(f.getPrerequisites());
      if (!annotation_features.isEmpty()) {
        scheduler.schedule(AnnotationFeatureJob.class);
        OutputJob.addPrerequisite(AnnotationFeatureJob.class);
      }
      scheduler.schedule(OutputJob.class);

      scheduler.percolate(OutputJob.class);

      // Paraphrase grammar mode.
    } else if ("paraphrasing".equals(type)) {
      // Schedule rule extraction job.
      scheduler.schedule(ExtractionJob.class);
      // Collect the translation grammar features required to compute
      // the requested paraphrasing features.
      Set<String> prereq_features = new HashSet<String>();
      List<PivotedFeature> pivoted_features = PivotedFeatureFactory.getAll(features);
      for (PivotedFeature pf : pivoted_features) {
        prereq_features.addAll(pf.getPrerequisites());
      }
      // Next, schedule translation features and register with feature
      // collection job.
      boolean annotation_features = false;
      for (String f_name : prereq_features) {
        MapReduceFeature mf = MapReduceFeatureFactory.get(f_name);
        if (mf != null) {
          scheduler.schedule(mf.getClass());
          FeatureCollectionJob.addPrerequisite(mf.getClass());
        } else {
          AnnotationFeature af = AnnotationFeatureFactory.get(f_name);
          if (af != null) {
            AnnotationFeatureJob.addPrerequisites(af.getPrerequisites());
            annotation_features = true;
          }
        }
      }
      if (annotation_features) {
        scheduler.schedule(AnnotationFeatureJob.class);
        FeatureCollectionJob.addPrerequisite(AnnotationFeatureJob.class);
      }
      scheduler.schedule(FeatureCollectionJob.class);
      // Schedule pivoting and pivoted feature computation job.
      scheduler.schedule(ParaphrasePivotingJob.class);
      // Schedule aggregation and output job.
      scheduler.schedule(ParaphraseAggregationJob.class);
      scheduler.percolate(ParaphraseAggregationJob.class);
    } else if ("distributional".equals(type)) {
      scheduler.schedule(DistributionalContextExtractionJob.class);
      scheduler.schedule(DistributionalContextSortingJob.class);
      scheduler.percolate(DistributionalContextSortingJob.class);
    } else {
      System.err.println("Unknown grammar type. No jobs scheduled.");
    }
  }

  public static void main(String[] argv) throws Exception {
    ToolRunner.run(null, new Thrax(), argv);
    return;
  }

  protected synchronized void workerDone(Class<? extends ThraxJob> theClass, boolean success) {
    try {
      scheduler.setState(theClass, success ? JobState.SUCCESS : JobState.FAILED);
    } catch (SchedulerException e) {
      System.err.println(e.getMessage());
    }
    notify();
    return;
  }

  public class ThraxJobWorker implements Runnable {
    private Thrax thrax;
    private Class<? extends ThraxJob> theClass;

    public ThraxJobWorker(Thrax t, Class<? extends ThraxJob> c, Configuration conf) {
      thrax = t;
      theClass = c;
    }

    public void run() {
      try {
        ThraxJob thraxJob = theClass.newInstance();
        Job job = thraxJob.getJob(conf);
        job.waitForCompletion(false);
        thrax.workerDone(theClass, job.isSuccessful());
      } catch (Exception e) {
        e.printStackTrace();
        thrax.workerDone(theClass, false);
      }
      return;
    }
  }
}
