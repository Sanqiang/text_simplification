package edu.jhu.thrax.distributional;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;

public class ContextPhrase {

  private final Text phrase;

  private MapWritable features;

  public ContextPhrase(String phrase) {
    this.phrase = new Text(phrase);
    this.features = new MapWritable();
  }

  public void addFeature(String feature_name) {
    addFeature(feature_name, 1);
  }

  public void addFeature(String feature_name, int feature_value) {
    Text feature_text = new Text(feature_name);
    Writable current_value = features.get(feature_text);
    if (current_value != null)
      features.put(feature_text, new IntWritable(((IntWritable) current_value).get()
          + feature_value));
    else
      features.put(feature_text, new IntWritable(feature_value));
  }

  public Text getPhrase() {
    return phrase;
  }

  public MapWritable getFeatures() {
    return features;
  }
}
