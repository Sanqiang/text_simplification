package edu.jhu.thrax.distributional;

import edu.jhu.thrax.distributional.FeatureTypes.Directionality;
import edu.jhu.thrax.distributional.FeatureTypes.Flavor;
import edu.jhu.thrax.distributional.FeatureTypes.Label;
import edu.jhu.thrax.distributional.FeatureTypes.Type;

public class FeatureEncoder {

  public static long encode(Type type, Label label, Flavor flavor, Directionality directionality) {
    return 0;
  }
  
  public static String type(long coded) {
    int feature_code = (int) (coded >> 32);
    
    return new Integer(feature_code).toString();
  }
  
  public static int feature(long coded) {
    return (int) (coded & 0x00000000FFFFFFFF);
  }
  
}
