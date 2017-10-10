package edu.jhu.thrax.distributional;

import edu.jhu.thrax.distributional.FeatureTypes.Label;
import edu.jhu.thrax.distributional.FeatureTypes.Type;

public class FeatureClass {
  public final Type type;
  public final Label label;
  public final int max_context;
  public final int max_gram;
  
  public FeatureClass(Type type, Label label) {
    this(type, label, -1, -1);
  }
  
  public FeatureClass(Type type, Label label, int max_context, int max_gram) {
    this.type = type;
    this.label = label;
    this.max_context = max_context;
    this.max_gram = max_gram;
  }
}