package edu.jhu.thrax.syntax;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EmptyStackException;
import java.util.Iterator;
import java.util.List;
import java.util.Scanner;
import java.util.Stack;

import edu.jhu.thrax.util.Vocabulary;

public class ParseTree {
  private final int[] labels;
  private final int[] numChildren;
  private final int[] start;
  private final int[] end;

  private ParseTree(int[] ls, int[] cs, int[] ss, int[] es) {
    labels = ls;
    numChildren = cs;
    start = ss;
    end = es;
  }

  public static ParseTree fromPennFormat(String s) {
    List<Integer> ls = new ArrayList<Integer>();
    List<Integer> cs = new ArrayList<Integer>();
    List<Integer> ss = new ArrayList<Integer>();
    List<Integer> es = new ArrayList<Integer>();
    try {
      buildLists(ls, cs, ss, es, s);
    } catch (EmptyStackException e) {
      return null;
    }
    int size = ls.size();
    int[] labels = new int[size];
    for (int i = 0; i < labels.length; ++i)
     labels[i] = ls.get(i);
    int[] numChildren = toIntArray(cs);
    int[] start = toIntArray(ss);
    int[] end = toIntArray(es);
    return new ParseTree(labels, numChildren, start, end);
  }

  private static int[] toIntArray(List<Integer> list) {
    int[] result = new int[list.size()];
    for (int i = 0; i < result.length; i++)
      result[i] = list.get(i);
    return result;
  }

  private static void addToken(List<Integer> ls, List<Integer> cs, List<Integer> ss,
      List<Integer> es, int label, int start, int end) {
    ls.add(label);
    cs.add(0);
    ss.add(start);
    es.add(end);
  }
  
  private static void increment(List<Integer> list, int index) {
    list.set(index, list.get(index) + 1);
  }

  private static void buildLists(List<Integer> ls, List<Integer> cs, List<Integer> ss,
      List<Integer> es, String line) {
    String input = line.trim();
    Stack<Integer> ancestors = new Stack<Integer>();

    int count = 0;
    int from = 0, to = 0;
    boolean seeking = true;
    boolean nonterminal = false;
    char current;
    // Run through entire (potentially parsed) sentence.
    while (from < input.length() && to < input.length()) {
      if (seeking) {
        current = input.charAt(from);
        if (current == '(') {
          ++from;
          nonterminal = true;
        } else if (current == ')') {
          ++from;
          ancestors.pop();
        } else if (current == ' ') {
          ++from;
        } else {
          to = from + 1;
          seeking = false;
        }
      } else {
        current = input.charAt(to);
        if (current == ' ' || current == ')' || current == '(') {
          if (nonterminal) {
            String nt = input.substring(from, to);
            if (nt.equals(",")) nt = "COMMA";
            int token = Vocabulary.id("[" + nt + "]");
            nonterminal = false;
            addToken(ls, cs, ss, es, token, count, count);
            if (!ancestors.empty()) increment(cs, ancestors.peek());
            ancestors.push(ls.size() - 1);
          } else {
            int token = Vocabulary.id(input.substring(from, to));
            addToken(ls, cs, ss, es, token, count, count + 1);
            for (int i : ancestors)
              increment(es, i);
            if (!ancestors.empty()) increment(cs, ancestors.peek());
            count++;
          }
          from = to;
          seeking = true;
        } else {
          ++to;
        }
      }
    }
  }

  public Node root() {
    return new Node(0);
  }

  public int numLeaves() {
    return end[0];
  }

  public int numNodes() {
    return labels.length;
  }

  public List<Node> internalNodesWithSpan(int from, int to) {
    int index = firstIndexOf(start, from);
    if (index < 0) return Collections.<Node>emptyList();
    List<Node> result = new ArrayList<Node>();
    while (index < start.length && start[index] == from && end[index] >= to) {
      if (end[index] == to && numChildren[index] > 0) result.add(new Node(index));
      index++;
    }
    return result;
  }

  private static int firstIndexOf(int[] array, int key) {
    int result = Arrays.binarySearch(array, key);
    if (result < 0) return result;
    while (result >= 0 && array[result] == key)
      result--;
    return result + 1;
  }

  public String toString() {
    return root().toString();
  }

  public boolean equals(Object o) {
    if (o == this) return true;
    if (!(o instanceof ParseTree)) return false;
    ParseTree other = (ParseTree) o;
    return Arrays.equals(labels, other.labels) && Arrays.equals(numChildren, other.numChildren)
        && Arrays.equals(start, other.start) && Arrays.equals(end, other.end);
  }

  public int hashCode() {
    int result = 163;
    result = result * 37 + Arrays.hashCode(labels);
    result = result * 37 + Arrays.hashCode(numChildren);
    result = result * 37 + Arrays.hashCode(start);
    result = result * 37 + Arrays.hashCode(end);
    return result;
  }

  public class Node {
    private final int index;

    public Node(int i) {
      index = i;
    }

    public int label() {
      return labels[index];
    }

    public int numChildren() {
      return numChildren[index];
    }

    public int spanStart() {
      return start[index];
    }

    public int spanEnd() {
      return end[index];
    }

    public Iterator<Node> children() {
      return new ChildIterator(index);
    }

    public String toString() {
      if (isLeaf()) return Vocabulary.word(label());
      String result = String.format("(%s", label());
      Iterator<Node> children = children();
      while (children.hasNext())
        result += " " + children.next().toString();
      result += ")";
      return result;
    }

    public boolean isLeaf() {
      return numChildren() == 0;
    }
  }

  private class ChildIterator implements Iterator<Node> {
    private final int index;
    private final int totalChildren;
    private int childrenSeen;
    private int childIndex;

    public ChildIterator(int i) {
      index = i;
      totalChildren = numChildren[index];
      childrenSeen = 0;
      childIndex = index + 1;
    }

    public boolean hasNext() {
      return childrenSeen < totalChildren;
    }

    public Node next() {
      Node result = new Node(childIndex);
      childIndex = nextSiblingIndex(childIndex);
      childrenSeen++;
      return result;
    }

    public void remove() {
      throw new UnsupportedOperationException();
    }

    private int nextSiblingIndex(int i) {
      int result = i + 1;
      int children = numChildren[i];
      for (int j = 0; j < children; j++)
        result = nextSiblingIndex(result);
      return result;
    }
  }

  public static void main(String[] argv) throws IOException {
    Scanner scanner = new Scanner(System.in);
    while (scanner.hasNextLine()) {
      ParseTree tree = ParseTree.fromPennFormat(scanner.nextLine());
      System.out.printf("%s\t%d\n", tree, tree.hashCode());
    }
    scanner.close();
  }
}
