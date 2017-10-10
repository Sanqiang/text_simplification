package edu.jhu.thrax.hadoop.extraction;

import org.apache.hadoop.io.Text;

import edu.jhu.thrax.hadoop.datatypes.AlignmentWritable;
import edu.jhu.thrax.hadoop.datatypes.Annotation;
import edu.jhu.thrax.hadoop.datatypes.RuleWritable;

public interface RuleWritableExtractor {
  public Iterable<AnnotatedRule> extract(Text line);
}


class AnnotatedRule {
  public RuleWritable rule = null;
  public AlignmentWritable f2e = null;
  public Annotation annotation = null;

  public AnnotatedRule(RuleWritable r) {
    rule = r;
  }

  public AnnotatedRule(RuleWritable r, AlignmentWritable f2e, Annotation a) {
    this.rule = r;
    this.f2e = f2e;
    this.annotation = a;
  }
}
