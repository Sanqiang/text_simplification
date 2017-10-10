package edu.jhu.thrax.hadoop.jobs;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;

import edu.jhu.thrax.util.FormatUtils;

public class Scheduler {
  private HashSet<String> faked;
  private HashMap<Class<? extends ThraxJob>, JobState> jobs;

  public Scheduler(Configuration config) {
    jobs = new HashMap<Class<? extends ThraxJob>, JobState>();
    faked = new HashSet<String>();

    String faked_line = config.get("thrax.fake");
    if (faked_line != null) {
      String[] faked_jobs = FormatUtils.P_SPACE.split(faked_line);
      for (String faked_job : faked_jobs)
        faked.add(faked_job);
    }
  }

  public boolean schedule(Class<? extends ThraxJob> jobClass) throws SchedulerException {
    if (jobs.containsKey(jobClass)) return false;
    ThraxJob job;
    try {
      job = jobClass.newInstance();
    } catch (Exception e) {
      e.printStackTrace();
      throw new SchedulerException(e.getMessage());
    }
    for (Class<? extends ThraxJob> c : job.getPrerequisites()) {
      schedule(c);
    }
    jobs.put(jobClass, JobState.PLANNED);
    System.err.println("[SCHED] planned job for " + jobClass);
    return true;
  }

  public boolean setState(Class<? extends ThraxJob> job_class, JobState state)
      throws SchedulerException {
    if (jobs.containsKey(job_class)) {
      jobs.put(job_class, state);
      System.err.println(String.format("[SCHED] %s in state %s", job_class, state));
      updateAllStates();
      return true;
    }
    return false;
  }

  @SuppressWarnings("fallthrough")
  public void updateAllStates() throws SchedulerException {
    for (Class<? extends ThraxJob> c : jobs.keySet()) {
      JobState state = jobs.get(c);
      switch (state) {
        case WAITING:
          checkReady(c);
          // fall through
        case READY:
          checkFailedPrereq(c);
          // fall through
        default:
          // do nothing
      }
    }
  }

  public void percolate(Class<? extends ThraxJob> job_class) throws SchedulerException {
    ThraxJob job;
    try {
      job = job_class.newInstance();
    } catch (Exception e) {
      throw new SchedulerException(e.getMessage());
    }
    Set<Class<? extends ThraxJob>> prereqs = job.getPrerequisites();

    if (faked.contains(job.getName())) {
      setState(job_class, JobState.SUCCESS);
    } else {
      setState(job_class, JobState.WAITING);
      if (prereqs != null) for (Class<? extends ThraxJob> p : prereqs)
        percolate(p);
    }
  }

  public void checkReady(Class<? extends ThraxJob> c) throws SchedulerException {
    ThraxJob job;
    try {
      job = c.newInstance();
    } catch (Exception e) {
      throw new SchedulerException(e.getMessage());
    }
    // check all succeeded
    // if state changes, have to recall check all states
    for (Class<? extends ThraxJob> p : job.getPrerequisites()) {
      if (!jobs.get(p).equals(JobState.SUCCESS)) return;
    }
    // All prereqs are in state SUCCESS.
    setState(c, JobState.READY);
  }

  public void checkFailedPrereq(Class<? extends ThraxJob> c) throws SchedulerException {
    ThraxJob job;
    try {
      job = c.newInstance();
    } catch (Exception e) {
      throw new SchedulerException(e.getMessage());
    }
    // check all succeeded
    // if state changes, have to recall check all states
    for (Class<? extends ThraxJob> p : job.getPrerequisites()) {
      JobState state = jobs.get(p);
      if (state.equals(JobState.FAILED) || state.equals(JobState.PREREQ_FAILED)) {
        setState(c, JobState.PREREQ_FAILED);
        return;
      }
    }
    return;
  }

  public JobState getState(Class<? extends ThraxJob> jobClass) {
    return jobs.get(jobClass);
  }

  public boolean isScheduled(Class<? extends ThraxJob> jobClass) {
    return jobs.containsKey(jobClass);
  }

  public Set<Class<? extends ThraxJob>> getClassesByState(JobState state) {
    Set<Class<? extends ThraxJob>> result = new HashSet<Class<? extends ThraxJob>>();
    for (Class<? extends ThraxJob> c : jobs.keySet()) {
      if (jobs.get(c).equals(state)) result.add(c);
    }
    return result;
  }

  public int numJobs() {
    return jobs.size();
  }

  public boolean notFinished() {
    for (Class<? extends ThraxJob> c : jobs.keySet()) {
      JobState state = jobs.get(c);
      if (state.equals(JobState.READY) || state.equals(JobState.WAITING)
          || state.equals(JobState.RUNNING)) return true;
    }
    return false;
  }

  public String toString() {
    StringBuilder sb = new StringBuilder();
    for (Class<? extends ThraxJob> c : jobs.keySet()) {
      sb.append(c + "\t" + jobs.get(c));
      sb.append("\n");
    }
    return sb.toString();
  }
}
