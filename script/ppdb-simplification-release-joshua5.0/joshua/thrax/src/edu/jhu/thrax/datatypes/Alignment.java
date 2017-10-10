package edu.jhu.thrax.datatypes;

import java.util.Iterator;

/**
 * This interface represents a word-level alignment of a sentence pair.
 */
public interface Alignment {

  public boolean sourceIndexIsAligned(int i);

  public boolean targetIndexIsAligned(int i);

  public int numTargetWordsAlignedTo(int i);

  public int numSourceWordsAlignedTo(int i);

  public Iterator<Integer> targetIndicesAlignedTo(int i);

  public Iterator<Integer> sourceIndicesAlignedTo(int i);

  public boolean consistentWith(int sourceLength, int targetLength);
}
