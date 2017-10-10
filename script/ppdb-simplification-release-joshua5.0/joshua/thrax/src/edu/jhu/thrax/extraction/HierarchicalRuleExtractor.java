package edu.jhu.thrax.extraction;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import edu.jhu.thrax.datatypes.Alignment;
import edu.jhu.thrax.datatypes.ArrayAlignment;
import edu.jhu.thrax.datatypes.HierarchicalRule;
import edu.jhu.thrax.datatypes.PhrasePair;
import edu.jhu.thrax.util.FormatUtils;
import edu.jhu.thrax.util.Vocabulary;

public class HierarchicalRuleExtractor {
  private int arityLimit = 2;
  private int initialPhraseSourceLimit = 10;
  private int initialPhraseTargetLimit = 10;
  private boolean requireMinimalPhrases = true;
  private int minimumInitialAlignmentPoints = 1;
  private boolean allowAdjacent = false;
  private int sourceSymbolLimit = 5;
  private int targetSymbolLimit = 1000;
  private int minimumRuleAlignmentPoints = 1;
  private int lexicalRuleSourceLimit = 10;
  private int lexicalRuleTargetLimit = 10;
  private boolean allowAbstract = false;
  private boolean allowMixed = true;
  private boolean allowFullSentenceRules = true;
  private PhrasePair fullSentencePhrasePair;

  public HierarchicalRuleExtractor() {
    // just use the defaults!
  }

  public HierarchicalRuleExtractor(int arity, int initialPhraseSource, int initialPhraseTarget,
      int initialAlignment, boolean initialAligned, int sourceLimit, int targetLimit,
      int ruleAlignment, boolean adjacent, boolean allow_abstract, boolean allow_mixed,
      boolean allowFullSentence, int lexSourceLimit, int lexTargetLimit) {
    arityLimit = arity;
    initialPhraseSourceLimit = initialPhraseSource;
    initialPhraseTargetLimit = initialPhraseTarget;
    minimumInitialAlignmentPoints = initialAlignment;
    requireMinimalPhrases = initialAligned;
    sourceSymbolLimit = sourceLimit;
    targetSymbolLimit = targetLimit;
    minimumRuleAlignmentPoints = ruleAlignment;
    lexicalRuleSourceLimit = lexSourceLimit;
    lexicalRuleTargetLimit = lexTargetLimit;
    allowAdjacent = adjacent;
    allowAbstract = allow_abstract;
    allowMixed = allow_mixed;
    allowFullSentenceRules = allowFullSentence;
  }

  public List<HierarchicalRule> extract(int sourceLength, int targetLength, Alignment alignment) {
    fullSentencePhrasePair = new PhrasePair(0, sourceLength, 0, targetLength);
    List<PhrasePair> initialPhrasePairs = initialPhrasePairs(sourceLength, targetLength, alignment);

    HierarchicalRule[][] rulesByArity = new HierarchicalRule[arityLimit + 1][];
    // we have one arity-0 rule for each initial phrase pair
    rulesByArity[0] = new HierarchicalRule[initialPhrasePairs.size()];
    for (int i = 0; i < initialPhrasePairs.size(); i++)
      rulesByArity[0][i] = new HierarchicalRule(initialPhrasePairs.get(i));
    // rules of arity j can be created from rules of arity j - 1 using the
    // initial phrase pairs
    for (int j = 1; j <= arityLimit; j++) {
      rulesByArity[j] = addNonterminalsTo(rulesByArity[j - 1], initialPhrasePairs);
    }
    return removeIfNotValid(rulesByArity, alignment);
  }

  private List<PhrasePair> initialPhrasePairs(int sourceLength, int targetLength, Alignment a) {
    List<PhrasePair> result = new ArrayList<PhrasePair>();
    for (int i = 0; i < sourceLength; i++) {
      for (int x = 1; x <= initialPhraseSourceLimit; x++) {
        if (i + x > sourceLength) break;
        for (int j = 0; j < targetLength; j++) {
          for (int y = 1; y <= initialPhraseTargetLimit; y++) {
            if (j + y > targetLength) break;
            PhrasePair pp = new PhrasePair(i, i + x, j, j + y);
            if (pp.isInitialPhrasePair(a, !requireMinimalPhrases, minimumInitialAlignmentPoints)) {
              result.add(pp);
            }
          }
        }
      }
      if (i == 0 && allowFullSentenceRules
          && !result.contains(fullSentencePhrasePair)) {
        result.add(fullSentencePhrasePair);
      }
    }
    return result;
  }

  private HierarchicalRule[] addNonterminalsTo(HierarchicalRule[] rules,
      List<PhrasePair> initialPhrasePairs) {
    List<HierarchicalRule> result = new ArrayList<HierarchicalRule>();
    for (HierarchicalRule r : rules) {
      int start = getStart(r, allowAdjacent);
      int end = r.getLhs().sourceEnd;
      for (PhrasePair pp : initialPhrasePairs) {
        if (pp.sourceStart < start) continue;
        if (pp.sourceStart >= end) break;
        if (r.getLhs().contains(pp)) {
          boolean disjointFromAllNTs = true;
          for (int i = 0; i < r.arity(); i++) {
            if (!r.getNonterminal(i).targetIsDisjointFrom(pp)) {
              disjointFromAllNTs = false;
              break;
            }
          }
          if (disjointFromAllNTs) result.add(r.addNonterminal(pp));
        }
      }
    }
    HierarchicalRule[] resultArray = new HierarchicalRule[result.size()];
    return result.toArray(resultArray);
  }

  private static int getStart(HierarchicalRule r, boolean allowAdjacent) {
    int arity = r.arity();
    if (arity == 0) return r.getLhs().sourceStart;
    int start = r.getNonterminal(arity - 1).sourceEnd;
    int offset = allowAdjacent ? 0 : 1;
    return start + offset;
  }

  private List<HierarchicalRule> removeIfNotValid(HierarchicalRule[][] rules, Alignment a) {
    List<HierarchicalRule> result = new ArrayList<HierarchicalRule>();
    for (HierarchicalRule[] rs : rules) {
      for (HierarchicalRule r : rs) {
        if (isValid(r, a)) result.add(r);
      }
    }
    return result;
  }

  private boolean isValid(HierarchicalRule r, Alignment a) {
    int arity = r.arity();
    int numSourceTerminals = r.numSourceTerminals();
    int numTargetTerminals = r.numTargetTerminals();
    // Conditions:
    // 1) rule size limits
    if (arity > 0) {
      // When we are a hierarchical rule,
      // 1a) limit of the total number of symbols on the source side
      // 1b) limit of the total number of symbols on the target side
      if (arity + numSourceTerminals > sourceSymbolLimit) return false;
      if (arity + numTargetTerminals > targetSymbolLimit) return false;
    } else {
      // When we are a lexical rule,
      // 1c) use lexical rule limits (even if full sentence rules
      // are allowed, we don't want to extract some giant 50-word phrase)
      if (numSourceTerminals > lexicalRuleSourceLimit) return false;
      if (numTargetTerminals > lexicalRuleTargetLimit) return false;
    }
    // 2) minimum number of alignment points
    if (r.numAlignmentPoints(a) < minimumRuleAlignmentPoints) return false;
    // 3) whether to allow abstract rules (with no terminals)
    if (!allowAbstract && numSourceTerminals == 0 && numTargetTerminals == 0) return false;
    // 4) whether to allow mixed rules (with NTs and terminals together)
    if (!allowMixed && arity > 0 && (numSourceTerminals > 0 || numTargetTerminals > 0))
      return false;
    // This is where you add more conditions!
    return true;
  }

  public static void main(String[] argv) throws IOException {
    Scanner scanner = new Scanner(System.in, "utf-8");
    HierarchicalRuleExtractor extractor = new HierarchicalRuleExtractor();
    SpanLabeler labeler = null;
    if (argv.length > 0) {
      if (argv[0].equals("--hiero")) labeler = new HieroLabeler(Vocabulary.id("X"));
    }
    while (scanner.hasNextLine()) {
      String line = scanner.nextLine();
      String[] parts = FormatUtils.P_DELIM.split(line);
      if (parts.length >= 3) {
        int[] source = Vocabulary.addAll(parts[0]);
        int[] target = Vocabulary.addAll(parts[1]);
        Alignment alignment = ArrayAlignment.fromString(parts[2], false);
        for (HierarchicalRule r : extractor.extract(source.length, target.length, alignment)) {
          if (labeler != null)
            System.out.println(r.toString(source, target, labeler, true));
          else
            System.out.println(r);
        }
      }
    }
    scanner.close();
    return;
  }

}
