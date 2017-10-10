package edu.jhu.thrax.hadoop.features;

import java.util.Map;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;

import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.util.Vocabulary;

public class WordCompressionRatioFeature implements SimpleFeature {

  public static final String NAME = "word-cr";
  public static final String LABEL = "WordLogCR";

  private static final IntWritable ZERO = new IntWritable(0);

  public Writable score(RuleWritable r) {
    int src_count = 0;
    for (int tok : r.source)
      if (!Vocabulary.nt(tok)) src_count++;
    int tgt_count = 0;
    for (int tok : r.target)
      if (!Vocabulary.nt(tok)) tgt_count++;
    if (src_count == 0 || tgt_count == 0) {
      return ZERO;
    } else {
      return new FloatWritable((float) Math.log((float) tgt_count / src_count));
    }
  }

  public String getName() {
    return NAME;
  }

  public String getLabel() {
    return LABEL;
  }

  public void unaryGlueRuleScore(int nt, Map<Integer, Writable> map) {
    map.put(Vocabulary.id(LABEL), ZERO);
  }

  public void binaryGlueRuleScore(int nt, Map<Integer, Writable> map) {
    map.put(Vocabulary.id(LABEL), ZERO);
  }
}
