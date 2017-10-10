package edu.jhu.thrax.hadoop.features.annotation;

import java.io.IOException;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Reducer.Context;

import edu.jhu.thrax.hadoop.datatypes.AlignmentWritable;
import edu.jhu.thrax.hadoop.datatypes.Annotation;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.hadoop.jobs.SourceWordGivenTargetWordProbabilityJob;
import edu.jhu.thrax.hadoop.jobs.ThraxJob;
import edu.jhu.thrax.lexprob.TrieLexprobTable;
import edu.jhu.thrax.util.Vocabulary;

@SuppressWarnings("rawtypes")
public class SourceGivenTargetLexicalProbabilityFeature implements AnnotationFeature {

  public static final String NAME = "f_given_e_lex";
  public static final String LABEL = "Lex(f|e)";
  
  private static final float DEFAULT_PROB = 10e-7f;
  
  private TrieLexprobTable table;
  
  public String getName() {
    return NAME;
  }
  
  public String getLabel() {
    return LABEL;
  }

  public void init(Context context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    String work_dir = conf.getRaw("thrax.work-dir");
    String sgt_path = work_dir + "lexprobs_sgt/part-*";
    table = new TrieLexprobTable(conf, sgt_path);
    context.progress();
  }

  public Writable score(RuleWritable key, Annotation annotation) {
    return new FloatWritable(sourceGivenTarget(key, annotation.f2e()));
  }

  private float sourceGivenTarget(RuleWritable rule, AlignmentWritable f2e) {
    byte[] points = f2e.points;
    int[] source = rule.source;
    int[] target = rule.target;
    
    float total = 0, prob = 0;
    int prev = -1;
    int n = points.length / 2;
    int m = 0;
    int expected = 0;
    for (int i = 0; i < n; ++i) {
      int f = points[2 * i];
      int e = points[2 * i + 1];

      if (f != prev && prev != -1) {
        total += Math.log(prob) - Math.log(m);
        prob = 0;
        m = 0;
      }
      prev = f;
      m++;

      while (expected < f) {
        if (!Vocabulary.nt(source[expected])) {
          float p = table.get(Vocabulary.getUnknownId(), source[expected]);
          total += (p < 0 ? Math.log(DEFAULT_PROB) : Math.log(p));
        }
        ++expected;
      }
      expected = f + 1;

      float p = table.get(target[e], source[f]);
      prob += (p < 0 ? DEFAULT_PROB : p);
      if (p < 0)
        System.err.printf("WARNING: could not read lexprob p(%s|%s)\n", Vocabulary.word(source[f]),
            Vocabulary.word(target[e]));
    }
    if (m != 0)
      total += Math.log(prob) - Math.log(m);
    
    while (expected < source.length) {
      if (!Vocabulary.nt(source[expected])) {
        float p = table.get(Vocabulary.getUnknownId(), source[expected]);
        total += (p < 0 ? Math.log(DEFAULT_PROB) : Math.log(p));
      }
      ++expected;
    }
    return -total;
  }


  public Set<Class<? extends ThraxJob>> getPrerequisites() {
    Set<Class<? extends ThraxJob>> pqs = new HashSet<Class<? extends ThraxJob>>();
    pqs.add(SourceWordGivenTargetWordProbabilityJob.class);
    return pqs;
  }

  private static final FloatWritable ONE_PROB = new FloatWritable(0.0f);

  public void unaryGlueRuleScore(int nt, Map<Integer, Writable> map) {
    map.put(Vocabulary.id(LABEL), ONE_PROB);
  }

  public void binaryGlueRuleScore(int nt, Map<Integer, Writable> map) {
    map.put(Vocabulary.id(LABEL), ONE_PROB);
  }
}
