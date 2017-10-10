package edu.jhu.thrax.lexprob;


/**
 * A data structure holding word-level lexical probabilities. The table only needs to support two
 * operations: determining whether a particular pair is present in the table, and returning the
 * probability associated with the pair.
 */
public interface LexicalProbabilityTable {
  /**
   * Return the lexical probability of a source language word given a target language word.
   * 
   * @param source the source language word
   * @param target the target language word
   * @return the probability p(source|target) if present, -1 otherwise
   */
  public float logpSourceGivenTarget(int source, int target);

  // TODO: these don't actually return -logp, they return p.

  /**
   * Return the lexical probability of a target language word given a source language word.
   * 
   * @param source the source language word
   * @param target the target language word
   * @return the probability p(target|source) is present, -1 otherwise
   */
  public float logpTargetGivenSource(int source, int target);
}
