package edu.jhu.thrax.lexprob;

import java.io.IOException;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;

public class HashMapLexprobTable extends SequenceFileLexprobTable {
  private HashMap<Long, Float> table;

  public HashMapLexprobTable(Configuration conf, String fileGlob) throws IOException {
    super(conf, fileGlob);
    Iterable<TableEntry> entries = getSequenceFileIterator(fs, conf, files);
    initialize(entries);
  }

  public void initialize(Iterable<TableEntry> entries) {
    table = new HashMap<Long, Float>();
    for (TableEntry te : entries) {
      table.put((((long) te.car << 32) | te.cdr), te.probability);
      if (table.size() % 1000 == 0) System.err.printf("[%d]\n", table.size());
    }
  }

  public float get(int car, int cdr) {
    long pair = (((long) car << 32) | cdr);
    if (table.containsKey(pair)) return table.get(pair);
    return -1.0f;
  }

  public boolean contains(int car, int cdr) {
    long pair = (((long) car << 32) | cdr);
    return table.containsKey(pair);
  }
}
