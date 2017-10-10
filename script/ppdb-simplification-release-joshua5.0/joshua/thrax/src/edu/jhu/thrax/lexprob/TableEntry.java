package edu.jhu.thrax.lexprob;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;

import edu.jhu.thrax.util.Vocabulary;

public class TableEntry {
  
  public final int car;
  public final int cdr;
  public final float probability;

  public TableEntry(LongWritable pair, FloatWritable d) {
    int first = (int) (pair.get() >> 32); 
    car = (first < 0 ? Vocabulary.getUnknownId() : first);
    cdr = (int) pair.get();
    probability = d.get();
  }

  public String toString() {
    return String.format("(%s,%s):%.4f", car, cdr, probability);
  }

  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof TableEntry)) return false;
    TableEntry te = (TableEntry) o;
    return car == te.car && cdr == te.cdr && probability == te.probability;
  }
}
