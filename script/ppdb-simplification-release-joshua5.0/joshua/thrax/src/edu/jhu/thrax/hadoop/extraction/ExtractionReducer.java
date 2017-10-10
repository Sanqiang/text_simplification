package edu.jhu.thrax.hadoop.extraction;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Reducer;

import edu.jhu.thrax.hadoop.datatypes.AlignedRuleWritable;
import edu.jhu.thrax.hadoop.datatypes.AlignmentWritable;
import edu.jhu.thrax.hadoop.datatypes.Annotation;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.util.Vocabulary;

public class ExtractionReducer
    extends Reducer<AlignedRuleWritable, Annotation, RuleWritable, Annotation> {

  private RuleWritable currentRule = null;
  private Annotation currentAnnotation = null;
  private AlignmentWritable maxAlignment = null;
  private int alignmentCount;

  private int minCount;

  protected void setup(Context context) throws IOException, InterruptedException {
    Configuration conf = context.getConfiguration();
    String vocabulary_path = conf.getRaw("thrax.work-dir") + "vocabulary/part-*";
    Vocabulary.initialize(conf, vocabulary_path);
    minCount = conf.getInt("thrax.min-rule-count", 1);
  }

  protected void reduce(AlignedRuleWritable key, Iterable<Annotation> values, Context context)
      throws IOException, InterruptedException {
    RuleWritable rule = key.getRule();
    AlignmentWritable alignment = key.getAlignment();

    Annotation merged = new Annotation();
    for (Annotation a : values)
      merged.merge(a);

    if (!rule.equals(currentRule)) {
      if (currentRule != null
          && (currentAnnotation.count() >= minCount || isUnigramRule(currentRule))) {
        currentAnnotation.setAlignment(maxAlignment);
        context.write(currentRule, currentAnnotation);
        context.progress();
      }
      currentRule = new RuleWritable(rule);
      currentAnnotation = new Annotation();
      alignmentCount = 0;
      maxAlignment = null;
    }
    currentAnnotation.merge(merged);
    if (alignmentCount < merged.count()) {
      maxAlignment = new AlignmentWritable(alignment);
      alignmentCount = merged.count();
    }
  }

  protected void cleanup(Context context) throws IOException, InterruptedException {
    if (currentRule != null) {
      if (currentAnnotation.count() >= minCount || isUnigramRule(currentRule)) {
        currentAnnotation.setAlignment(maxAlignment);
        context.write(currentRule, currentAnnotation);        
        context.progress();
      }
    }
  }

  private static boolean isUnigramRule(RuleWritable rule) {
    if (rule.source.length == 1) return !Vocabulary.nt(rule.source[0]);
    return rule.target.length == 1 && !Vocabulary.nt(rule.target[0]);
  }
}
