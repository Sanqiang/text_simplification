package edu.jhu.thrax.extraction;

import java.util.HashMap;

import edu.jhu.thrax.util.Vocabulary;

public enum LabelCache {
  SLASH("/"), BACKSLASH("\\"), PLUS("+");
  
  private HashMap<Long, Integer> cache = new HashMap<Long, Integer>();
  private String glue;
  
  private LabelCache(String g) {
    glue = g;
  }
  
  public final int get(int left, int right) {
    long key = ((long) left << 32) | ((long) right & 0x00000000FFFFFFFFL);
    Integer val = cache.get(key);
    if (val == null) {
      val = join(left, right, glue);
      cache.put(key, val);
    }
    return val;
  }
  
  private static final int join(int a, int b, String glue) {
    String word_a = Vocabulary.word(a);
    String word_b = Vocabulary.word(b);
    return Vocabulary.id(word_a.substring(0, word_a.length() - 1) + glue
        + word_b.substring(1));
  }
}