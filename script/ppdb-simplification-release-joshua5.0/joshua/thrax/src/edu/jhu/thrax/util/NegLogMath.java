package edu.jhu.thrax.util;

public class NegLogMath {

  // Number of entries in the table.
  private static final int LOG_ADD_TABLE_SIZE = 640000;
  // Smallest value for nlog_a - nlog_b.
  private static final float LOG_ADD_MIN = -64.0f;
  private static final float AS_GOOD_AS_ZERO = 1e-10f;
  private static final float logAddInc = -LOG_ADD_MIN / LOG_ADD_TABLE_SIZE;
  private static final float invLogAddInc = LOG_ADD_TABLE_SIZE / -LOG_ADD_MIN;
  private static final float[] logAddTable = new float[LOG_ADD_TABLE_SIZE + 1];

  static {
    for (int i = 0; i <= LOG_ADD_TABLE_SIZE; i++) {
      logAddTable[i] = (float) -Math.log1p(Math.exp((i * logAddInc) + LOG_ADD_MIN));
    }
  }

  public static float logAdd(float nlog_a, float nlog_b) {
    if (nlog_b < nlog_a) {
      float temp = nlog_a;
      nlog_a = nlog_b;
      nlog_b = temp;
    }
    float neg_diff = (nlog_a - nlog_b) - LOG_ADD_MIN;
    if (neg_diff < AS_GOOD_AS_ZERO) {
      return nlog_a;
    }
    return nlog_a + logAddTable[(int) (neg_diff * invLogAddInc)];
  }
}
