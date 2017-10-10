package edu.jhu.thrax.datatypes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import edu.jhu.thrax.util.FormatUtils;

public class ArrayAlignment implements Alignment {
  private final int[] sourceIndicesInOrder;
  private final int[] targetIndicesSourceOrder;

  private final int[] targetIndicesInOrder;
  private final int[] sourceIndicesTargetOrder;

  private ArrayAlignment(List<IntPair> pairs, boolean reverse) {
    int[] sortedCars = cars(pairs);
    int[] cdrs = cdrs(pairs);
    List<IntPair> reversedPairs = reverseAll(pairs);
    Collections.sort(reversedPairs);
    int[] sortedCdrs = cars(reversedPairs);
    int[] cars = cdrs(reversedPairs);
    if (!reverse) {
      sourceIndicesInOrder = sortedCars;
      targetIndicesSourceOrder = cdrs;
      targetIndicesInOrder = sortedCdrs;
      sourceIndicesTargetOrder = cars;
    } else {
      sourceIndicesInOrder = sortedCdrs;
      targetIndicesSourceOrder = cars;
      targetIndicesInOrder = sortedCars;
      sourceIndicesTargetOrder = cdrs;
    }
  }

  public static ArrayAlignment fromString(String s, boolean reverse) {
    String[] tokens = FormatUtils.P_SPACE.split(s);
    List<IntPair> pairs = getIntPairsFromTokens(tokens);
    Collections.sort(pairs);
    return new ArrayAlignment(pairs, reverse);
  }

  public boolean consistentWith(int sourceLength, int targetLength) {
    if (sourceIndicesInOrder.length == 0) // alignment is empty
      return true;
    return sourceIndicesInOrder[0] >= 0 && targetIndicesInOrder[0] >= 0
        && sourceIndicesInOrder[sourceIndicesInOrder.length - 1] < sourceLength
        && targetIndicesInOrder[targetIndicesInOrder.length - 1] < targetLength;
  }

  public boolean sourceIndexIsAligned(int i) {
    return Arrays.binarySearch(sourceIndicesInOrder, i) >= 0;
  }

  public boolean targetIndexIsAligned(int i) {
    return Arrays.binarySearch(targetIndicesInOrder, i) >= 0;
  }

  public int numTargetWordsAlignedTo(int i) {
    return frequency(sourceIndicesInOrder, i);
  }

  public int numSourceWordsAlignedTo(int i) {
    return frequency(targetIndicesInOrder, i);
  }

  public Iterator<Integer> targetIndicesAlignedTo(int i) {
    int start = firstIndexOf(sourceIndicesInOrder, i);
    if (start < 0) return Collections.<Integer>emptyList().iterator();
    return new AlignmentIterator(sourceIndicesInOrder, targetIndicesSourceOrder, i, start);
  }

  public Iterator<Integer> sourceIndicesAlignedTo(int i) {
    int start = firstIndexOf(targetIndicesInOrder, i);
    if (start < 0) return Collections.<Integer>emptyList().iterator();
    return new AlignmentIterator(targetIndicesInOrder, sourceIndicesTargetOrder, i, start);
  }
  
  public byte[] toCompactSourceArray() {
    byte[] result = new byte[sourceIndicesInOrder.length * 2];
    for (int i = 0; i < sourceIndicesInOrder.length; ++i) {
      result[i * 2] = (byte) sourceIndicesInOrder[i];
      result[i * 2 + 1] = (byte) targetIndicesSourceOrder[i];
    }
    return result;
  }
  
  public byte[] toCompactTargetArray() {
    byte[] result = new byte[targetIndicesInOrder.length * 2];
    for (int i = 0; i < targetIndicesInOrder.length; ++i) {
      result[i * 2] = (byte) targetIndicesInOrder[i];
      result[i * 2 + 1] = (byte) sourceIndicesTargetOrder[i];
    }
    return result;
  }

  private static int[] cars(List<IntPair> list) {
    int[] result = new int[list.size()];
    for (int i = 0; i < result.length; i++)
      result[i] = list.get(i).fst;
    return result;
  }

  private static int[] cdrs(List<IntPair> list) {
    int[] result = new int[list.size()];
    for (int i = 0; i < result.length; i++)
      result[i] = list.get(i).snd;
    return result;
  }

  private static List<IntPair> reverseAll(List<IntPair> list) {
    List<IntPair> result = new ArrayList<IntPair>(list.size());
    for (int i = 0; i < list.size(); i++)
      result.add(i, list.get(i).reverse());
    return result;
  }

  private static List<IntPair> getIntPairsFromTokens(String[] toks) {
    List<IntPair> result = new ArrayList<IntPair>();
    for (String t : toks) {
      IntPair ip = IntPair.fromHyphenatedString(t);
      if (ip != null) result.add(ip);
    }
    return result;
  }

  private static int firstIndexOf(int[] array, int key) {
    int index = Arrays.binarySearch(array, key);
    if (index < 0) return index;
    while (index >= 0 && array[index] == key)
      index--;
    return index + 1;
  }

  private static int lastIndexOf(int[] array, int key) {
    int index = Arrays.binarySearch(array, key);
    if (index < 0) return index;
    while (index < array.length && array[index] == key)
      index++;
    return index;
  }

  private static int frequency(int[] array, int key) {
    int start = firstIndexOf(array, key);
    if (start < 0) return 0;
    int end = lastIndexOf(array, key);
    return end - start;
  }
}


class AlignmentIterator implements Iterator<Integer> {
  private final int[] keys;
  private final int[] values;
  private final int key;
  private int current;

  public AlignmentIterator(int[] ks, int[] vs, int k, int start) {
    keys = ks;
    values = vs;
    key = k;
    current = start;
  }

  public boolean hasNext() {
    return current < keys.length && keys[current] == key;
  }

  public Integer next() {
    int result = values[current];
    current++;
    return result;
  }

  public void remove() {
    throw new UnsupportedOperationException();
  }
}
