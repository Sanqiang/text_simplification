package edu.jhu.thrax.hadoop.features;

import java.util.Map;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;

import edu.jhu.thrax.hadoop.datatypes.RuleWritable;
import edu.jhu.thrax.util.Vocabulary;

public class CharacterCountDifferenceFeature implements SimpleFeature {

  private static final IntWritable ZERO = new IntWritable(0);

  public static final String NAME = "char-count-difference";
  public static final String LABEL = "CharCountDiff";
  
  public Writable score(RuleWritable r) {
    int char_difference = 0;
    for (int tok : r.source) {
      if (!Vocabulary.nt(tok)) {
        char_difference -= Vocabulary.word(tok).length();
      }
    }
    char_difference -= r.source.length - 1;

    for (int tok : r.target) {
      if (!Vocabulary.nt(tok)) {
        char_difference += Vocabulary.word(tok).length();
      }
    }
    char_difference += r.target.length - 1;
    return new IntWritable(char_difference);
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
