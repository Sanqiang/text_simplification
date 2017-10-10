package edu.jhu.thrax.hadoop.datatypes;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.WritableUtils;

/**
 * This is a utility class. It provides read/write implementations around arrays of primitives
 * (e.g., int[], long[], etc.), that vaguely resemble Writable structures. There is no
 * de-serialization safety provided: code using these methods is responsible for using matching
 * read/write calls.
 */
public final class PrimitiveUtils {

  // TODO: re-add variable-length stuff

  public static final int MARGINAL_ID = 0;

  public static final int compare(byte a, byte b) {
    return a - b;
  }

  public static final int compare(int a, int b) {
    return a - b;
  }
  
  public static final int compare(boolean a, boolean b) {
    return (a == b ? 0 : (a ? 1 : -1));
  }

  public static final int compareIntArrays(int[] a, int[] b) {
    for (int i = 0; i < Math.min(a.length, b.length); ++i) {
      if (a[i] < b[i]) {
        return -1;
      } else if (a[i] > b[i]) {
        return 1;
      }
    }
    if (a.length < b.length) return -1;
    if (a.length > b.length) return 1;
    return 0;
  }
  
  public static final int compareByteArrays(byte[] a, byte[] b) {
    for (int i = 0; i < Math.min(a.length, b.length); ++i) {
      if (a[i] < b[i]) {
        return -1;
      } else if (a[i] > b[i]) {
        return 1;
      }
    }
    if (a.length < b.length) return -1;
    if (a.length > b.length) return 1;
    return 0;
  }
  
  public static final void writeBooleanArray(DataOutput out, boolean[] array) throws IOException {
    WritableUtils.writeVInt(out, array.length);
    for (boolean b : array)
      out.writeBoolean(b);
  }

  public static final void writeCharArray(DataOutput out, char[] array) throws IOException {
    WritableUtils.writeVInt(out, array.length * 2);
    for (char b : array)
      out.writeChar(b);
  }

  public static final void writeByteArray(DataOutput out, byte[] array) throws IOException {
    WritableUtils.writeVInt(out, array.length);
    out.write(array, 0, array.length);
  }

  public static final void writeShortArray(DataOutput out, short[] array) throws IOException {
    WritableUtils.writeVInt(out, array.length * 2);
    for (short b : array)
      out.writeShort(b);
  }

  public static final void writeIntArray(DataOutput out, int[] array) throws IOException {
    WritableUtils.writeVInt(out, array.length * 4);
    for (int b : array)
      out.writeInt(b);
  }

  public static final void writeLongArray(DataOutput out, long[] array) throws IOException {
    WritableUtils.writeVInt(out, array.length * 8);
    for (long b : array)
      out.writeLong(b);
  }

  public static final void writeFloatArray(DataOutput out, float[] array) throws IOException {
    WritableUtils.writeVInt(out, array.length * 4);
    for (float b : array)
      out.writeFloat(b);
  }

  public static final void writeDoubleArray(DataOutput out, double[] array) throws IOException {
    WritableUtils.writeVInt(out, array.length * 8);
    for (double b : array)
      out.writeDouble(b);
  }

  public static final boolean[] readBooleanArray(DataInput in) throws IOException {
    boolean[] array = new boolean[WritableUtils.readVInt(in)];
    for (int i = 0; i < array.length; i++)
      array[i] = in.readBoolean();
    return array;
  }

  public static final char[] readCharArray(DataInput in) throws IOException {
    char[] array = new char[WritableUtils.readVInt(in) / 2];
    for (int i = 0; i < array.length; i++)
      array[i] = in.readChar();
    return array;
  }

  public static final byte[] readByteArray(DataInput in) throws IOException {
    byte[] array = new byte[WritableUtils.readVInt(in)];
    in.readFully(array, 0, array.length);
    return array;
  }

  public static final short[] readShortArray(DataInput in) throws IOException {
    short[] array = new short[WritableUtils.readVInt(in) / 2];
    for (int i = 0; i < array.length; i++)
      array[i] = in.readShort();
    return array;
  }

  public static final int[] readIntArray(DataInput in) throws IOException {
    int[] array = new int[WritableUtils.readVInt(in) / 4];
    for (int i = 0; i < array.length; i++)
      array[i] = in.readInt();
    return array;
  }

  public static final long[] readLongArray(DataInput in) throws IOException {
    long[] array = new long[WritableUtils.readVInt(in) / 8];
    for (int i = 0; i < array.length; i++)
      array[i] = in.readLong();
    return array;
  }

  public static final float[] readFloatArray(DataInput in) throws IOException {
    float[] array = new float[WritableUtils.readVInt(in) / 4];
    for (int i = 0; i < array.length; i++)
      array[i] = in.readFloat();
    return array;
  }

  public static final double[] readDoubleArray(DataInput in) throws IOException {
    double[] array = new double[WritableUtils.readVInt(in) / 8];
    for (int i = 0; i < array.length; i++)
      array[i] = in.readDouble();
    return array;
  }
}
