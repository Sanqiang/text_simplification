package edu.jhu.thrax.hadoop.datatypes;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

public class AlignmentWritable implements WritableComparable<AlignmentWritable> {

  public static final Text SGT_KEY = new Text("e2f_align");
  public static final Text TGS_KEY = new Text("f2e_align");

  public byte[] points;

  // Cached target-to-source alignment.
  private AlignmentWritable flipped = null;

  public AlignmentWritable() {
    points = new byte[0];
  }

  public AlignmentWritable(byte[] p) {
    points = p;
  }

  public AlignmentWritable(byte[] p, float[] c) {
    points = p;
  }

  public AlignmentWritable(byte[] p, float[] c, int t) {
    points = p;
  }

  public AlignmentWritable(AlignmentWritable r) {
    this.set(r);
  }

  public void set(AlignmentWritable r) {
    points = Arrays.copyOf(r.points, r.points.length);
  }

  public AlignmentWritable flip() {
    if (flipped == null) {
      Integer[] flipside_points = new Integer[points.length / 2];
      for (int i = 0; i < flipside_points.length; ++i)
        flipside_points[i] = i;
      Arrays.sort(flipside_points, new Comparator<Integer>() {
        public int compare(Integer a, Integer b) {
          return PrimitiveUtils.compare(points[2 * a + 1], points[2 * b + 1]);
        }
      });

      byte[] flipside = new byte[points.length];
      for (int i = 0; i < flipside_points.length; ++i) {
        flipside[2 * i] = points[2 * flipside_points[i] + 1];
        flipside[2 * i + 1] = points[2 * flipside_points[i]];
      }
      flipped = new AlignmentWritable(flipside);
    }
    return flipped;
  }

  public AlignmentWritable join(AlignmentWritable that) {
    ArrayList<Byte> joined = new ArrayList<Byte>();
    int j = 0;
    int j_max = that.points.length / 2;
    for (int i = 0; i < this.points.length / 2; ++i) {
      byte through = this.points[2 * i];
      byte from = this.points[2 * i + 1];
      while (j < j_max && that.points[2 * j] < through)
        j++;
      int start = j;
      while (j < j_max && that.points[2 * j] == through) {
        joined.add(from);
        joined.add(that.points[2 * j + 1]);
        j++;
      }
      // Jump to start of this through-point in "that", in case the next through point is the same.
      j = start;
    }
    byte[] join_points = new byte[joined.size()];
    for (int i = 0; i < join_points.length; ++i)
      join_points[i] = joined.get(i);
    return new AlignmentWritable(join_points);
  }

  public AlignmentWritable intersect(AlignmentWritable that) {
    ArrayList<Byte> common = new ArrayList<Byte>();

    int j = 0;
    int j_max = that.points.length / 2;
    for (int i = 0; i < this.points.length / 2; ++i) {
      byte from = this.points[2 * i];
      byte to = this.points[2 * i + 1];
      while (j < j_max && that.points[2 * j] < from)
        ++j;
      if (j < j_max && that.points[2 * j] != from) continue;
      int start = j;
      while (j < j_max && that.points[2 * j] == from && that.points[2 * j + 1] != to)
        ++j;
      if (j < j_max && that.points[2 * j] == from && that.points[2 * j + 1] == to) {
        common.add(from);
        common.add(to);
      }
      j = start;
    }
    byte[] common_points = new byte[common.size()];
    for (int i = 0; i < common_points.length; ++i)
      common_points[i] = common.get(i);
    return new AlignmentWritable(common_points);
  }

  public void write(DataOutput out) throws IOException {
    PrimitiveUtils.writeByteArray(out, points);
  }

  public void readFields(DataInput in) throws IOException {
    points = PrimitiveUtils.readByteArray(in);
  }

  public String toString(String glue) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < points.length / 2; ++i) {
      if (i != 0) sb.append(glue);
      sb.append(points[2 * i]);
      sb.append("-");
      sb.append(points[2 * i + 1]);
    }
    return sb.toString();
  }

  public String toString() {
    return toString(" ");
  }

  public int compareTo(AlignmentWritable that) {
    return PrimitiveUtils.compareByteArrays(this.points, that.points);
  }

  static {
    WritableComparator.define(AlignmentWritable.class, new AlignmentComparator());
  }

  public static final class AlignmentComparator extends WritableComparator {

    public AlignmentComparator() {
      super(AlignmentWritable.class);
    }

    public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) {
      return WritableComparator.compareBytes(b1, s1, l1, b2, s2, l2);
    }
  }
}
