package edu.jhu.thrax.hadoop.datatypes;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableUtils;

public class Annotation implements Writable {

  // Source-to-target alignment.
  private AlignmentWritable f2e = null;

  // Rule occurrence count.
  private int count;

  public Annotation() {
    count = 0;
  }

  public Annotation(int c) {
    count = c;
  }

  public Annotation(Annotation a) {
    count = a.count;
    this.f2e = new AlignmentWritable(a.f2e);
  }
  
  public Annotation(AlignmentWritable f2e) {
    count = 1;
    this.f2e = f2e;
  }

  public void merge(Annotation that) {
    this.count += that.count;
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    boolean has_alignments = false;
    count = WritableUtils.readVInt(in);
    if (count < 0) {
      count = -count;
      has_alignments = true;
    }
    if (has_alignments) {
      f2e = new AlignmentWritable();
      f2e.readFields(in);
    }
  }

  @Override
  public void write(DataOutput out) throws IOException {
    WritableUtils.writeVInt(out, (f2e != null ? -count : count));
    if (f2e != null) f2e.write(out);
  }

  public AlignmentWritable e2f() {
      return f2e.flip();
  }

  public AlignmentWritable f2e() {
    return f2e;
  }

  public void setAlignment(AlignmentWritable a) {
    f2e = a;
  }

  public int count() {
    return count;
  }
}
