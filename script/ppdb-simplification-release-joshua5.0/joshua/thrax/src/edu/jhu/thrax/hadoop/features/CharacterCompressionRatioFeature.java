package edu.jhu.thrax.hadoop.features;

import java.util.Map;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Writable;

import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.util.Vocabulary;

public class CharacterCompressionRatioFeature implements SimpleFeature {

  private static final FloatWritable ZERO = new FloatWritable(0f);

  public static final String NAME = "char-cr";
  public static final String LABEL = "CharLogCR";
  
  public Writable score(RuleWritable r) {
    int src_length = 0;
    for (int tok : r.source) {
      if (!Vocabulary.nt(tok)) {
        src_length += Vocabulary.word(tok).length();
      }
    }
    src_length += r.source.length - 1;

    int tgt_length = 0;
    for (int tok : r.target) {
      if (!Vocabulary.nt(tok)) {
        tgt_length += Vocabulary.word(tok).length();
      }
    }
    tgt_length += r.target.length - 1;

    if (src_length == 0 || tgt_length == 0)
      return ZERO;
    else
      return new FloatWritable((float) Math.log((float) tgt_length / src_length));
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
