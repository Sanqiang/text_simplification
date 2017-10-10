package edu.jhu.thrax.lexprob;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.conf.Configuration;

public class TrieLexprobTable extends SequenceFileLexprobTable {
  private int[] cars;
  private int[][] cdrs;
  private float[][] values;
  
  public TrieLexprobTable(Configuration conf, String fileGlob) throws IOException {
    super(conf, fileGlob);
    Iterable<TableEntry> entries = getSequenceFileIterator(fs, conf, files);
    int size = getNumCars(entries);
    cars = new int[size];
    cdrs = new int[size][];
    values = new float[size][];
    entries = getSequenceFileIterator(fs, conf, files);
    initialize(entries);
  }

  private static int getNumCars(Iterable<TableEntry> entries) {
    int result = 0;
    int prev = -1;
    for (TableEntry te : entries) {
      if (te.car != prev) {
        result++;
        prev = te.car;
      }
    }
    return result;
  }

  protected void initialize(Iterable<TableEntry> entries) {
    int i = 0;
    int car = -1;
    List<Integer> cdrList = new ArrayList<Integer>();
    List<Float> valueList = new ArrayList<Float>();
    for (TableEntry te : entries) {
      if (car == -1) {
        car = te.car;
        cars[i] = car;
      }
      if (te.car != car) {
        cdrs[i] = intArray(cdrList);
        values[i] = floatArray(valueList);
        cdrList.clear();
        valueList.clear();
        i++;
        if (i % 1000 == 0) System.err.printf("[%d]\n", i);
        cars[i] = te.car;
        car = cars[i];
      }
      cdrList.add(te.cdr);
      valueList.add(te.probability);
    }
    cdrs[i] = intArray(cdrList);
    values[i] = floatArray(valueList);
  }

  private static float[] floatArray(List<Float> list) {
    float[] result = new float[list.size()];
    for (int i = 0; i < list.size(); i++)
      result[i] = list.get(i);
    return result;
  }

  private static int[] intArray(List<Integer> list) {
    int[] result = new int[list.size()];
    for (int i = 0; i < list.size(); i++)
      result[i] = list.get(i);
    return result;
  }

  public float get(int car, int cdr) {
    int i = Arrays.binarySearch(cars, car);
    if (i < 0) // the car is not present
      return 0;
    int j = Arrays.binarySearch(cdrs[i], cdr);
    if (j < 0) // the cdr is not present
      return 0;
    return values[i][j];
  }

  public boolean contains(int car, int cdr) {
    int i = Arrays.binarySearch(cars, car);
    if (i < 0) return false;
    return Arrays.binarySearch(cdrs[i], cdr) >= 0;
  }
}
