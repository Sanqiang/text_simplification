package edu.jhu.thrax.tools;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.logging.Logger;

import edu.jhu.jerboa.util.FileManager;
import edu.jhu.thrax.util.io.LineReader;

public class JudgeParaphrases {

  private static final Logger logger = Logger.getLogger(JudgeParaphrases.class.getName());

  public static void main(String[] args) {

    String input = null;
    String output = null;

    for (int i = 0; i < args.length; i++) {
      if ("-i".equals(args[i]) && (i < args.length - 1)) {
        input = args[++i];
      } else if ("-o".equals(args[i]) && (i < args.length - 1)) {
        output = args[++i];
      }
    }

    if (input == null) {
      logger.severe("No input file specified.");
      return;
    }
    if (output == null) {
      logger.severe("No output file specified.");
      return;
    }

    LineReader reader = null;
    BufferedWriter writer = null;
    Scanner user = null;
    try {
      reader = new LineReader(input);
      writer = FileManager.getWriter(output);
      user = new Scanner(System.in);
      while (reader.hasNext()) {
        String pp = reader.next().trim();
        System.out.print(pp + "\t");
        String score = user.next().trim();
        if (score.toLowerCase().equals("quit") || score.toLowerCase().equals("exit"))
          break;
        writer.write(score + "\t" + pp + "\n");
      }
      reader.close();
      writer.close();
    } catch (IOException e) {
      logger.severe(e.getMessage());
    }
  }

}
