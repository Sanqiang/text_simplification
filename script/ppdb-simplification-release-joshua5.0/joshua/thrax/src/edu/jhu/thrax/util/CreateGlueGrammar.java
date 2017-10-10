package edu.jhu.thrax.util;

import java.io.File;
import java.io.IOException;
import java.util.HashSet;

import edu.jhu.thrax.util.io.LineReader;

public class CreateGlueGrammar {
  private static HashSet<String> nts;

  // [GOAL] ||| [GOAL,1] [X,2] ||| [GOAL,1] [X,2] ||| -1
  // [GOAL] ||| [GOAL,1] </s> ||| [GOAL,1] </s> ||| 0
  // [GOAL] ||| <s> ||| <s> ||| 0

  private static final String R_START = "[%1$s] ||| <s> ||| <s> ||| 0";
  private static final String R_TWO = "[%1$s] ||| [%1$s,1] [%2$s,2] ||| [%1$s,1] [%2$s,2] ||| -1";
  private static final String R_END = "[%1$s] ||| [%1$s,1] </s> ||| [%1$s,1] </s> ||| 0";

  // [GOAL] ||| <s> [X,1] </s> ||| <s> [X,1] </s> ||| 0
  private static final String R_TOP = "[%1$s] ||| <s> [%2$s,1] </s> ||| <s> [%2$s,1] </s> ||| 0";

  private static String GOAL = "GOAL";

  public static void main(String[] argv) throws IOException {
    String grammar_file_name = null;
    if (argv.length > 0) grammar_file_name = argv[0];
    if (argv.length > 1) GOAL = argv[1];

    if (grammar_file_name == null) {
      System.err.println("No grammar specified.");
      System.exit(1);
    }
    File grammar_file = new File(grammar_file_name);
    if (!grammar_file.exists()) {
      System.err.println("Grammar file doesn't exist: " + grammar_file_name);
      System.exit(1);
    }

    nts = new HashSet<String>();
    if (grammar_file.isDirectory()) {
      Vocabulary.read(grammar_file_name + File.separator + "vocabulary");
      for (int i = 0; i < Vocabulary.size(); ++i) {
        String token = Vocabulary.word(i);
        if (Vocabulary.nt(token)) nts.add(token.substring(1, token.length() - 1));
      }
    } else {
      LineReader reader = new LineReader(grammar_file_name);
      while (reader.hasNext()) {
        String line = reader.next();
        int lhsStart = line.indexOf("[") + 1;
        int lhsEnd = line.indexOf("]");
        if (lhsStart < 1 || lhsEnd < 0) {
          System.err.printf("malformed rule: %s\n", line);
          continue;
        }
        String lhs = line.substring(lhsStart, lhsEnd);
        nts.add(lhs);
      }
    }

    System.out.println(String.format(R_START, GOAL));
    for (String nt : nts)
      System.out.println(String.format(R_TWO, GOAL, nt));
    System.out.println(String.format(R_END, GOAL));
    for (String nt : nts)
      System.out.println(String.format(R_TOP, GOAL, nt));

  }
}
