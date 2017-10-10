package edu.jhu.thrax.hadoop.paraphrasing;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.mapreduce.Reducer;

import edu.jhu.thrax.hadoop.datatypes.Annotation;
import edu.jhu.thrax.hadoop.datatypes.FeatureMap;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.hadoop.features.annotation.AnnotationFeature;
import edu.jhu.thrax.hadoop.features.annotation.AnnotationFeatureFactory;
import edu.jhu.thrax.hadoop.features.annotation.AnnotationPassthroughFeature;
import edu.jhu.thrax.hadoop.features.pivot.PivotedAnnotationFeature;
import edu.jhu.thrax.hadoop.features.pivot.PivotedFeature;
import edu.jhu.thrax.hadoop.features.pivot.PivotedFeatureFactory;
import edu.jhu.thrax.util.BackwardsCompatibility;
import edu.jhu.thrax.util.Vocabulary;

public class PivotingReducer extends Reducer<RuleWritable, FeatureMap, RuleWritable, FeatureMap> {

  private static enum PivotingCounters {
    F_READ, EF_READ, EF_PRUNED, EE_PRUNED, EE_WRITTEN
  }

  private int[] currentSrc;
  private int currentLhs;

  private int[] nts;
  private int lhs;

  private List<ParaphrasePattern> targets;
  private List<PivotedFeature> pivotedFeatures;
  private List<AnnotationFeature> annotationFeatures;

  private Map<Integer, PruningRule> translationPruningRules;
  private Map<Integer, PruningRule> pivotedPruningRules;

  protected void setup(Context context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    String vocabulary_path = conf.getRaw("thrax.work-dir") + "vocabulary/part-*";
    Vocabulary.initialize(conf, vocabulary_path);

    String features = BackwardsCompatibility.equivalent(conf.get("thrax.features", ""));
    pivotedFeatures = PivotedFeatureFactory.getAll(features);

    annotationFeatures = AnnotationFeatureFactory.getAll(features);
    if (!annotationFeatures.isEmpty()) pivotedFeatures.add(new PivotedAnnotationFeature());

    Set<String> prerequisite_features = new HashSet<String>();
    for (PivotedFeature pf : pivotedFeatures)
      prerequisite_features.addAll(pf.getPrerequisites());
    annotationFeatures = new ArrayList<AnnotationFeature>();
    for (String f_name : prerequisite_features) {
      AnnotationFeature af = AnnotationFeatureFactory.get(f_name);
      if (af != null) annotationFeatures.add(af);
    }

    currentLhs = 0;
    currentSrc = null;

    lhs = 0;
    nts = null;
    targets = new ArrayList<ParaphrasePattern>();

    String pruning_rules = BackwardsCompatibility.equivalent(conf.get("thrax.pruning", ""));
    translationPruningRules = getTranslationPruningRules(pruning_rules);
    pivotedPruningRules = getPivotedPruningRules(pruning_rules);

    for (AnnotationFeature af : annotationFeatures)
      af.init(context);
  }

  protected void reduce(RuleWritable key, Iterable<FeatureMap> values, Context context)
      throws IOException, InterruptedException {
    if (currentLhs == 0 || !(key.lhs == currentLhs && Arrays.equals(key.source, currentSrc))) {
      if (currentLhs != 0) pivotAll(context);
      currentLhs = key.lhs;
      currentSrc = key.source;
      // TODO: not sure why this check is here.
      if (currentLhs == 0 || currentSrc.length == 0) return;
      lhs = currentLhs;
      nts = extractNonterminals(currentSrc);
      targets.clear();
    }
    boolean seen_first = false;
    for (FeatureMap features : values) {
      if (seen_first)
        throw new RuntimeException("Multiple feature maps for one rule:" + key.toString());
      seen_first = true;

      Annotation annotation =
          (Annotation) features.get(AnnotationPassthroughFeature.LABEL);
      for (AnnotationFeature f : annotationFeatures)
        features.put(f.getLabel(), f.score(key, annotation));

      if (!prune(features, translationPruningRules))
        targets.add(new ParaphrasePattern(key.target, nts, lhs, key.monotone, features));
      else
        context.getCounter(PivotingCounters.EF_PRUNED).increment(1);
    }
  }

  protected void cleanup(Context context) throws IOException, InterruptedException {
    if (currentLhs != 0) pivotAll(context);
  }

  protected void pivotAll(Context context) throws IOException, InterruptedException {
    context.getCounter(PivotingCounters.F_READ).increment(1);
    context.getCounter(PivotingCounters.EF_READ).increment(targets.size());

    for (int i = 0; i < targets.size(); i++) {
      for (int j = i; j < targets.size(); j++) {
        pivotOne(targets.get(i), targets.get(j), context);
        if (i != j) pivotOne(targets.get(j), targets.get(i), context);
      }
    }
  }

  protected void pivotOne(ParaphrasePattern src, ParaphrasePattern tgt, Context context)
      throws IOException, InterruptedException {
    RuleWritable pivoted_rule = new RuleWritable();
    FeatureMap pivoted_features = new FeatureMap();

    pivoted_rule.lhs = src.lhs;
    pivoted_rule.source = src.rhs;
    pivoted_rule.target = tgt.rhs;
    pivoted_rule.monotone = (src.monotone == tgt.monotone);

    try {
      // Compute the features.
      for (PivotedFeature f : pivotedFeatures)
        pivoted_features.put(f.getLabel(), f.pivot(src.features, tgt.features));
    } catch (Exception e) {
      StringBuilder src_f = new StringBuilder();
      for (int w : src.features.keySet())
        src_f.append(Vocabulary.word(w) + "=" + src.features.get(w) + " ");
      StringBuilder tgt_f = new StringBuilder();
      for (int w : tgt.features.keySet())
        tgt_f.append(Vocabulary.word(w) + "=" + tgt.features.get(w) + " ");
      e.printStackTrace();
      throw new RuntimeException(Vocabulary.getWords(src.rhs) + " \n "
          + Vocabulary.getWords(tgt.rhs) + " \n " + src_f.toString() + " \n " + tgt_f.toString()
          + " \n");
    }


    if (!prune(pivoted_features, pivotedPruningRules)) {
      context.write(pivoted_rule, pivoted_features);
      context.getCounter(PivotingCounters.EE_WRITTEN).increment(1);
    } else {
      context.getCounter(PivotingCounters.EE_PRUNED).increment(1);
    }
  }

  protected Map<Integer, PruningRule> getPivotedPruningRules(String conf_string) {
    Map<Integer, PruningRule> rules = new HashMap<Integer, PruningRule>();
    // TODO: use patterns for this.
    String[] rule_strings = conf_string.split("\\s*,\\s*");
    for (String rule_string : rule_strings) {
      String[] f;
      boolean smaller;
      if (rule_string.contains("<")) {
        f = rule_string.split("<");
        smaller = true;
      } else if (rule_string.contains(">")) {
        f = rule_string.split(">");
        smaller = false;
      } else {
        continue;
      }
      int label = Vocabulary.id(PivotedFeatureFactory.get(f[0]).getLabel());
      rules.put(label, new PruningRule(smaller, Float.parseFloat(f[1])));
    }
    return rules;
  }

  protected Map<Integer, PruningRule> getTranslationPruningRules(String conf_string) {
    Map<Integer, PruningRule> rules = new HashMap<Integer, PruningRule>();
    String[] rule_strings = conf_string.split("\\s*,\\s*");
    for (String rule_string : rule_strings) {
      String[] f;
      boolean smaller;
      if (rule_string.contains("<")) {
        f = rule_string.split("<");
        smaller = true;
      } else if (rule_string.contains(">")) {
        f = rule_string.split(">");
        smaller = false;
      } else {
        continue;
      }
      Float threshold = Float.parseFloat(f[1]);
      Set<String> lower_bound_labels = PivotedFeatureFactory.get(f[0]).getLowerBoundLabels();
      if (lower_bound_labels != null) for (String label : lower_bound_labels)
        rules.put(Vocabulary.id(label), new PruningRule(smaller, threshold));

      Set<String> upper_bound_labels = PivotedFeatureFactory.get(f[0]).getUpperBoundLabels();
      if (upper_bound_labels != null) for (String label : upper_bound_labels)
        rules.put(Vocabulary.id(label), new PruningRule(!smaller, threshold));
    }
    return rules;
  }

  protected static boolean prune(FeatureMap features, final Map<Integer, PruningRule> rules) {
    for (Map.Entry<Integer, PruningRule> e : rules.entrySet()) {
      if (features.containsKey(e.getKey())
          && e.getValue().applies((FloatWritable) features.get(e.getKey()))) return true;
    }
    return false;
  }

  protected static int[] extractNonterminals(int[] source) {
    int first_nt = 0;
    for (int token : source)
      if (Vocabulary.nt(token)) {
        if (first_nt == 0)
          first_nt = token;
        else
          return new int[] {first_nt, token};
      }
    return (first_nt == 0 ? new int[0] : new int[] {first_nt});
  }

  class ParaphrasePattern {
    int arity;
    int lhs;
    int[] rhs;
    boolean monotone;

    FeatureMap features;

    public ParaphrasePattern(int[] target, int[] nts, int lhs, boolean mono, FeatureMap features) {
      this.arity = nts.length;

      this.lhs = lhs;
      this.rhs = target;
      this.monotone = mono;
      this.features = new FeatureMap(features);
    }
  }

  class PruningRule {
    private boolean smaller;
    private float threshold;

    PruningRule(boolean smaller, float threshold) {
      this.smaller = smaller;
      this.threshold = threshold;
    }

    protected boolean applies(FloatWritable value) {
      return (smaller ? value.get() < threshold : value.get() > threshold);
    }
  }
}
