package edu.jhu.thrax.hadoop.comparators;

import java.io.IOException;

import org.apache.hadoop.io.WritableComparator;
import org.apache.hadoop.io.WritableUtils;

public class FieldComparator {
  private final int fieldNumber;
  private final WritableComparator comparator;
  
  public int offset;

  public FieldComparator(int field, WritableComparator comparator) {
    if (field < 0)
      throw new IllegalArgumentException("TextFieldComparator: cannot compare field of index "
          + field);
    fieldNumber = field;
    this.comparator = comparator;
  }

  public int compare(byte[] b1, int s1, int l1, byte[] b2, int s2, int l2) throws IOException {
    int start1 = getFieldStart(fieldNumber, b1, s1);
    int start2 = getFieldStart(fieldNumber, b2, s2);

    int length1 = getFieldLength(b1, start1);
    int length2 = getFieldLength(b2, start2);

    // TODO: l1 and l2 may need to be adjusted to reflect offset.
    return comparator.compare(b1, start1, length1, b2, start2, length2);
  }

  private final int getFieldStart(int field, byte[] bytes, int start) throws IOException {
    // if we want the first field, just return current start
    if (field == 0) return start;
    // otherwise, find out how long this field is ...
    int fieldLength = getFieldLength(bytes, start);
    // then decrement the field number and find the next start
    return getFieldStart(field - 1, bytes, start + fieldLength);
  }

  private static final int getFieldLength(byte[] bytes, int start) throws IOException {
    // Text is serialized as vInt (the length) plus that many bytes
    int vint_size = WritableUtils.decodeVIntSize(bytes[start]);
    int field_length = WritableComparator.readVInt(bytes, start);
    return vint_size + field_length;
  }

  public int fieldEndIndex(byte[] bytes, int start) throws IOException {
    int fieldStart = getFieldStart(fieldNumber, bytes, start);
    int fieldLength = getFieldLength(bytes, fieldStart);
    return fieldStart + fieldLength;
  }
}
