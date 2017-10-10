package edu.jhu.thrax.distributional;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.TreeMap;

import org.apache.commons.lang3.StringEscapeUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;

import edu.jhu.thrax.syntax.LatticeArray;
import edu.jhu.thrax.util.FormatUtils;
import edu.jhu.thrax.util.Vocabulary;
import edu.jhu.thrax.util.exceptions.MalformedInputException;
import edu.jhu.thrax.util.exceptions.NotEnoughFieldsException;
import edu.jhu.thrax.util.io.LineReader;

@SuppressWarnings("unchecked")
public class ContextPhraseExtractor {

  private final String G = "_";

  private final String L = "l" + G;
  private final String C = "c" + G;
  private final String R = "r" + G;

  private final String LEX = "lex" + G;
  private final String POS = "pos" + G;
  private final String LEM = "lem" + G;
  private final String SYN = "syn" + G;
  private final String NER = "ner" + G;
  private final String DEP = "dep" + G;
  private final String GOV = "gov" + G;

  private final String ROOT = "ROOT";

  private int MAX_PHRASE_LENGTH;

  private final boolean USE_SYN;
  
  private int MAX_LEX_CONTEXT;
  private int MAX_POS_CONTEXT;
  private int MAX_LEM_CONTEXT;
  private int MAX_NER_CONTEXT;

  private int MAX_LEX_GRAM;
  private int MAX_POS_GRAM;
  private int MAX_LEM_GRAM;
  private int MAX_NER_GRAM;

  private final boolean USE_LEX;
  private final boolean USE_POS;
  private final boolean USE_LEM;
  private final boolean USE_NER;

  private final boolean READ_DEP;
  private final boolean USE_LEX_DEP;
  private final boolean USE_POS_DEP;
  private final boolean USE_LEM_DEP;
  private final boolean USE_NER_DEP;

  private final boolean READ_CDEP;
  private final boolean USE_LEX_CDEP;
  private final boolean USE_POS_CDEP;
  private final boolean USE_LEM_CDEP;
  private final boolean USE_NER_CDEP;

  private final boolean READ_CPDEP;
  private final boolean USE_LEX_CPDEP;
  private final boolean USE_POS_CPDEP;
  private final boolean USE_LEM_CPDEP;
  private final boolean USE_NER_CPDEP;

  private String[][] lex_features;
  private String[][] pos_features;
  private String[][] lem_features;
  private String[][] ner_features;

  private int size;

  private LatticeArray parse;
  private String[] lemma;
  private String[] ner;

  private ArrayList<Dependency>[] govern;
  private Dependency[] depend;
  private ArrayList<Dependency>[] c_govern;
  private Dependency[] c_depend;
  private ArrayList<Dependency>[] cp_govern;
  private Dependency[] cp_depend;

  public ContextPhraseExtractor(Configuration conf) {
    MAX_PHRASE_LENGTH = conf.getInt("thrax.max-phrase-length", 4);

    USE_SYN = conf.getBoolean("thrax.use-syn", false);
    
    MAX_LEX_CONTEXT = conf.getInt("thrax.max-lex-context", 4);
    MAX_POS_CONTEXT = conf.getInt("thrax.max-pos-context", 4);
    MAX_LEM_CONTEXT = conf.getInt("thrax.max-lem-context", 4);
    MAX_NER_CONTEXT = conf.getInt("thrax.max-ner-context", 4);

    MAX_LEX_GRAM = conf.getInt("thrax.max-lex-gram", 2);
    MAX_POS_GRAM = conf.getInt("thrax.max-pos-gram", 2);
    MAX_LEM_GRAM = conf.getInt("thrax.max-lem-gram", 2);
    MAX_NER_GRAM = conf.getInt("thrax.max-ner-gram", 2);

    USE_LEX = conf.getBoolean("thrax.use-lex-ngrams", false);
    USE_POS = conf.getBoolean("thrax.use-pos-ngrams", false);
    USE_LEM = conf.getBoolean("thrax.use-lem-ngrams", false);
    USE_NER = conf.getBoolean("thrax.use-ner-ngrams", false);

    USE_LEX_DEP = conf.getBoolean("thrax.use-lex-dep", false);
    USE_POS_DEP = conf.getBoolean("thrax.use-pos-dep", false);
    USE_LEM_DEP = conf.getBoolean("thrax.use-lem-dep", false);
    USE_NER_DEP = conf.getBoolean("thrax.use-ner-dep", false);
    READ_DEP = USE_LEX_DEP || USE_POS_DEP || USE_LEM_DEP || USE_NER_DEP;

    USE_LEX_CDEP = conf.getBoolean("thrax.use-lex-cdep", false);
    USE_POS_CDEP = conf.getBoolean("thrax.use-pos-cdep", false);
    USE_LEM_CDEP = conf.getBoolean("thrax.use-lem-cdep", false);
    USE_NER_CDEP = conf.getBoolean("thrax.use-ner-cdep", false);
    READ_CDEP = USE_LEX_CDEP || USE_POS_CDEP || USE_LEM_CDEP || USE_NER_CDEP;

    USE_LEX_CPDEP = conf.getBoolean("thrax.use-lex-cpdep", false);
    USE_POS_CPDEP = conf.getBoolean("thrax.use-pos-cpdep", false);
    USE_LEM_CPDEP = conf.getBoolean("thrax.use-lem-cpdep", false);
    USE_NER_CPDEP = conf.getBoolean("thrax.use-ner-cpdep", false);
    READ_CPDEP = USE_LEX_CPDEP || USE_POS_CPDEP || USE_LEM_CPDEP || USE_NER_CPDEP;
  }

  // Format is as follows:
  // parse ||| lemma ||| NER ||| basic deps ||| collapsed deps ||| colpp deps
  public List<ContextPhrase> extract(String input) throws MalformedInputException {
    List<ContextPhrase> output = new ArrayList<ContextPhrase>();
    try {
      input = StringEscapeUtils.unescapeXml(input);

      String[] inputs = FormatUtils.P_DELIM.split(input);
      if (inputs.length < 6) throw new NotEnoughFieldsException();

      parse = new LatticeArray(inputs[0].trim(), true);
      lemma = FormatUtils.P_SPACE.split(inputs[1].trim().toLowerCase());

      size = lemma.length;
      if (size != parse.size()) throw new MalformedInputException();

      String[] ner_entries = FormatUtils.P_SPACE.split(inputs[2].trim().toLowerCase());
      ner = new String[ner_entries.length];
      if (ner.length != size)
        throw new MalformedInputException("NER: " + ner.length + " vs. Size: " + size);
      for (int i = 0; i < ner_entries.length; ++i)
        ner[i] = FormatUtils.P_SLASH.split(ner_entries[i])[1];

      generateAllGramFeatures();

      if (READ_DEP) {
        govern = new ArrayList[size];
        depend = new Dependency[size];
        initDependencyStructure(inputs[3], govern, depend, USE_LEX_DEP, USE_LEM_DEP, USE_POS_DEP,
            USE_NER_DEP);
      }
      if (READ_CDEP) {
        c_govern = new ArrayList[size];
        c_depend = new Dependency[size];
        initDependencyStructure(inputs[4], c_govern, c_depend, USE_LEX_CDEP, USE_LEM_CDEP,
            USE_POS_CDEP, USE_NER_CDEP);
      }
      if (READ_CPDEP) {
        cp_govern = new ArrayList[size];
        cp_depend = new Dependency[size];
        initDependencyStructure(inputs[5], cp_govern, cp_depend, USE_LEX_CPDEP, USE_LEM_CPDEP,
            USE_POS_CPDEP, USE_NER_CPDEP);
      }

      for (int i = 0; i < size; i++) {
        for (int j = i + 1; j <= Math.min(i + MAX_PHRASE_LENGTH, size); j++) {
          ContextPhrase cp = new ContextPhrase(parse.getTerminalPhrase(i, j));
          if (USE_LEX) addGramFeatures(cp, i, j, MAX_LEX_CONTEXT, MAX_LEX_GRAM, LEX, lex_features);
          if (USE_POS) addGramFeatures(cp, i, j, MAX_POS_CONTEXT, MAX_POS_GRAM, POS, pos_features);
          if (USE_LEM) addGramFeatures(cp, i, j, MAX_LEM_CONTEXT, MAX_LEM_GRAM, LEM, lem_features);
          if (USE_NER) addGramFeatures(cp, i, j, MAX_NER_CONTEXT, MAX_NER_GRAM, NER, ner_features);
          if (USE_SYN) addSyntaxFeatures(i, j, cp);
          if (READ_DEP)
            addDependencyFeatures(i, j, cp, govern, depend, USE_LEX_DEP, USE_LEM_DEP, USE_POS_DEP,
                USE_NER_DEP);
          if (READ_CDEP)
            addDependencyFeatures(i, j, cp, c_govern, c_depend, USE_LEX_CDEP, USE_LEM_CDEP,
                USE_POS_CDEP, USE_NER_CDEP);
          if (READ_CPDEP)
            addDependencyFeatures(i, j, cp, cp_govern, cp_depend, USE_LEX_CPDEP, USE_LEM_CPDEP,
                USE_POS_CPDEP, USE_NER_CPDEP);
          output.add(cp);
        }
      }
    } catch (Exception e) {
      e.printStackTrace();
      throw new MalformedInputException();
    }
    return output;
  }

  private void generateAllGramFeatures() {
    if (USE_LEX) lex_features = buildGramFeatures(parse.getTerminals(), MAX_LEX_GRAM);
    if (USE_POS) pos_features = buildGramFeatures(parse.getPOS(), MAX_POS_GRAM);
    if (USE_LEM) lem_features = buildGramFeatures(lemma, MAX_LEM_GRAM);
    if (USE_NER) ner_features = buildGramFeatures(ner, MAX_NER_GRAM);
  }

  private String[][] buildGramFeatures(String[] sentence, int N) {
    String[][] cache = new String[size][];
    for (int i = 0; i <= size - N; i++)
      cache[i] = new String[N];
    for (int i = 1; i < N; i++)
      cache[size - N + i] = new String[N - i];

    StringBuilder sb = new StringBuilder();
    for (int cf = 0; cf < size; cf++) {
      sb.delete(0, sb.length());
      for (int l = 0; l < Math.min(N, size - cf); l++) {
        sb.append(sentence[cf + l]).append(G);
        cache[cf][l] = sb.toString();
      }
    }
    return cache;
  }

  private void addGramFeatures(ContextPhrase cp, int from, int to, int max_window, int N,
      String tag, String[][] cache) {
    String left_prefix = L + tag;
    for (int cf = Math.max(0, from - max_window); cf < from; cf++)
      for (int l = 0; l < Math.min(N, from - cf); l++)
        cp.addFeature(left_prefix + cache[cf][l] + (from - cf));

    String right_prefix = R + tag;
    final int right_boundary = Math.min(size, to + max_window);
    for (int cf = to; cf < right_boundary; cf++)
      for (int l = 0; l < Math.min(N, right_boundary - cf); l++)
        cp.addFeature(right_prefix + cache[cf][l] + (cf - to + 1));
  }

  private void addSyntaxFeatures(int from, int to, ContextPhrase cp) {
    Collection<Integer> constituents = parse.getConstituentLabels(from, to);
    for (int c : constituents)
      cp.addFeature(C + SYN + "span" + G + Vocabulary.word(c));

    Collection<Integer> ccg = parse.getCcgLabels(from, to);
    for (int c : ccg) {
      String label = Vocabulary.word(c);
      if (label.contains("/")) {
        String[] parts = FormatUtils.P_SLASH.split(label);
        cp.addFeature(R + SYN + "pref" + G + parts[0]);
        cp.addFeature(R + SYN + "miss" + G + parts[1]);
      } else {
        String[] parts = FormatUtils.P_BSLASH.split(label);
        cp.addFeature(L + SYN + "suff" + G + parts[0]);
        cp.addFeature(L + SYN + "miss" + G + parts[1]);
      }
    }
  }

  private void initDependencyStructure(String input, ArrayList<Dependency>[] gov, Dependency[] dep,
      boolean use_lex, boolean use_lem, boolean use_pos, boolean use_ner) {
    for (int i = 0; i < size; i++)
      gov[i] = new ArrayList<Dependency>();

    String[] entries = FormatUtils.P_SPACE.split(input.trim());
    for (String entry : entries) {
      Dependency d = new Dependency(entry, use_lex, use_lem, use_pos, use_ner);
      if (d.gov >= 0) gov[d.gov].add(d);
      dep[d.dep] = d;
    }
  }

  private void addDependencyFeatures(int from, int to, ContextPhrase cp,
      ArrayList<Dependency>[] gov, Dependency[] dep, boolean use_lex, boolean use_lem,
      boolean use_pos, boolean use_ner) {
    int head = from;
    boolean seen_outlink = false;
    boolean valid = true;
    for (int p = from; p < to; p++) {
      if (dep[p] != null) {
        if (dep[p].gov < from || dep[p].gov >= to) {
          dep[p].addDependingFeatures(cp, use_lex, use_lem, use_pos, use_ner);
          valid = valid && !seen_outlink;
          if (valid) head = p;
          seen_outlink = true;
        } else if (valid && p == head) {
          head = dep[p].gov;
        }
      } else if (gov[p].isEmpty()) {
        valid = false;
      }
      for (Dependency d : gov[p]) {
        if (d.dep < from || d.dep >= to) {
          d.addGoverningFeatures(cp, use_lex, use_lem, use_pos, use_ner);
          valid = false;
        }
      }
    }
    if (valid) {
      if (use_lex) cp.addFeature(C + "head" + G + LEX + parse.getTerminal(head));
      if (use_lem) cp.addFeature(C + "head" + G + LEM + lemma[head]);
      if (use_pos) cp.addFeature(C + "head" + G + POS + parse.getPOS(head));
      if (use_ner) cp.addFeature(C + "head" + G + NER + ner[head]);
    }
  }

  public static void main(String[] args) throws Exception {
    LineReader reader = new LineReader(args[0]);

    ContextPhraseExtractor cpe = new ContextPhraseExtractor(new Configuration());

    while (reader.hasNext()) {
      String line = reader.next().trim();
      List<ContextPhrase> cps = cpe.extract(line);
      for (ContextPhrase cp : cps) {
        TreeMap<Text, Integer> feature_map = new TreeMap<Text, Integer>();
        for (Writable fn : cp.getFeatures().keySet())
          feature_map.put((Text) fn, ((IntWritable) cp.getFeatures().get(fn)).get());
        System.out.println(FormatUtils.contextPhraseToText(cp.getPhrase(), feature_map));
      }
    }
  }

  class Dependency {
    final String type;
    final int gov;
    final int dep;

    final String dep_lex;
    final String gov_lex;
    final String dep_lem;
    final String gov_lem;
    final String dep_pos;
    final String gov_pos;
    final String dep_ner;
    final String gov_ner;

    public Dependency(String entry, boolean use_lex, boolean use_lem, boolean use_pos,
        boolean use_ner) {
      String[] fields = FormatUtils.P_DASH.split(entry);
      gov = Integer.parseInt(fields[1]) - 1;
      dep = Integer.parseInt(fields[0]) - 1;
      type = fields[2];

      String dep_side = (gov > dep ? R : L);
      String gov_side = (gov > dep ? L : R);

      if (use_lex) {
        dep_lex = dep_side + DEP + type + G + LEX + (gov == -1 ? ROOT : parse.getTerminal(gov));
        gov_lex = gov_side + GOV + type + G + LEX + parse.getTerminal(dep);
      } else {
        dep_lex = null;
        gov_lex = null;
      }
      if (use_pos) {
        dep_pos = dep_side + DEP + type + G + POS + (gov == -1 ? ROOT : parse.getPOS(gov));
        gov_pos = gov_side + GOV + type + G + POS + parse.getPOS(dep);
      } else {
        dep_pos = null;
        gov_pos = null;
      }
      if (use_lem) {
        dep_lem = dep_side + DEP + type + G + LEM + (gov == -1 ? ROOT : lemma[gov]);
        gov_lem = gov_side + GOV + type + G + LEM + lemma[dep];
      } else {
        dep_lem = null;
        gov_lem = null;
      }
      if (use_ner) {
        dep_ner = dep_side + DEP + type + G + NER + (gov == -1 ? ROOT : ner[gov]);
        gov_ner = gov_side + GOV + type + G + NER + ner[dep];
      } else {
        dep_ner = null;
        gov_ner = null;
      }
    }

    final void addGoverningFeatures(ContextPhrase cp, boolean use_lex, boolean use_lem,
        boolean use_pos, boolean use_ner) {
      if (use_lex) cp.addFeature(gov_lex);
      if (use_pos) cp.addFeature(gov_pos);
      if (use_lem) cp.addFeature(gov_lem);
      if (use_ner) cp.addFeature(gov_ner);
    }

    final void addDependingFeatures(ContextPhrase cp, boolean use_lex, boolean use_lem,
        boolean use_pos, boolean use_ner) {
      if (use_lex) cp.addFeature(dep_lex);
      if (use_pos) cp.addFeature(dep_pos);
      if (use_lem) cp.addFeature(dep_lem);
      if (use_ner) cp.addFeature(dep_ner);
    }
  }
}
