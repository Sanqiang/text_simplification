package edu.jhu.thrax.distributional;

import java.util.HashMap;
import java.util.Map;

public class FeatureTypes {

  public enum Type {
    NGRAM(0, "ngram"), SYN(1, "syn"), DEP(2, "dep"), CDEP(3, "cdep"), CPDEP(4, "cpdep");

    private static Map<Integer, Type> map;

    static {
      map = new HashMap<Integer, Type>();
      for (Type t : Type.values())
        map.put(t.code, t);
    }

    public static Type get(int code) {
      return map.get(code);
    }

    public final int code;
    public final String name;

    Type(int code, String name) {
      this.code = code;
      this.name = name;
    }
  }

  public enum Label {
    NONE(0, "none"), LEX(1, "lex"), LEM(2, "lem"), POS(3, "pos"), NER(4, "ner");

    public final int code;
    public final String name;

    Label(int code, String name) {
      this.code = code;
      this.name = name;
    }
  }

  public enum Directionality {
    NONE(0, "none"), LEFT(1, "left"), RIGHT(2, "right"), CENTER(3, "center");

    public final int code;
    public final String name;

    Directionality(int code, String name) {
      this.code = code;
      this.name = name;
    }
  }

  public enum Flavor {
    NONE(0, "none"), GOV(1, "gov"), DEP(2, "dep"), HEAD(3, "head");

    public final int code;
    public final String name;

    Flavor(int code, String name) {
      this.code = code;
      this.name = name;
    }
  }
}
