package edu.jhu.thrax.hadoop.features;

import java.util.Map;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Writable;

import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.util.Vocabulary;

public class WordLengthDifferenceFeature implements SimpleFeature {

  public static final String NAME = "word-length-difference";
  public static final String LABEL = "WordLenDiff";

  private static final FloatWritable ZERO = new FloatWritable(0);

  public Writable score(RuleWritable r) {
    int src_length = 0;
    int src_count = 0;
    for (int tok : r.source) {
      if (!Vocabulary.nt(tok)) {
        src_length += Vocabulary.word(tok).length();
        src_count++;
      }
    }
    int tgt_length = 0;
    int tgt_count = 0;
    for (int tok : r.target) {
      if (!Vocabulary.nt(tok)) {
        tgt_length += Vocabulary.word(tok).length();
        tgt_count++;
      }
    }
    if (src_count == 0 || tgt_count == 0) {
      return ZERO;
    } else {
      float avg_src_length = (float) src_length / src_count;
      float avg_tgt_length = (float) tgt_length / tgt_count;
      return new FloatWritable(avg_tgt_length - avg_src_length);
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
