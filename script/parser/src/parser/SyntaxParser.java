package parser;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

import javax.xml.bind.NotIdentifiableEvent;

import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.SentenceUtils;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.trees.Tree;

public class SyntaxParser {

	private String generate_tree(Tree node, StringBuilder sb_output) {
		if (node.children().length == 0) {
			return node.toString();
		}

		StringBuilder sb = new StringBuilder();
		for (Tree child_node : node.children()) {
			String output = generate_tree(child_node, sb_output);
			sb.append(output);
			sb.append(" ");
		}
		if (sb.toString().matches("[a-zA-Z ]+")) {
			String key = node.label().toString();
			String val = sb.toString().trim();
			sb_output.append(key).append("=>").append(val).append("\t");
		}
		return sb.toString();

	}

	public static void main(String[] args) throws Exception {
		String parserModel = "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz";
		LexicalizedParser lp = LexicalizedParser.loadModel(parserModel);
		
//		BufferedWriter writer = new BufferedWriter(new FileWriter(new File(
//				"/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src.jsyntax")));
//		BufferedReader reader = new BufferedReader(new FileReader(new File(
//				"/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikilarge/wiki.full.aner.train.src")));
		BufferedWriter writer = new BufferedWriter(new FileWriter(new File(
				"/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.train.src.jsyntax")));
		BufferedReader reader = new BufferedReader(new FileReader(new File(
				"/Users/zhaosanqiang916/git/text_simplification_data/train/dress/wikismall/PWKP_108016.tag.80.aner.train.src")));
		String line = null;
		SyntaxParser test = new SyntaxParser();
		int cnt = 0;
		StringBuilder output = new StringBuilder();
		long pre_time = System.currentTimeMillis();
		while (null != (line = reader.readLine())) {
			String[] sent = line.split(" ");
			List<CoreLabel> rawWords = SentenceUtils.toCoreLabelList(sent);
			Tree parse = lp.apply(rawWords);
			StringBuilder generate_out = new StringBuilder();
//			System.out.println(parse.toString());
			test.generate_tree(parse, generate_out);
			output.append(generate_out).append("\n");
//			System.out.println(generate_out);
			++cnt;
			if (cnt % 100 == 0) {
				long cur_time = System.currentTimeMillis() / 1000;
				long diff = cur_time - pre_time;
				System.out.println("Process idx " + cnt + " in " + diff);
				pre_time = cur_time;
			}
			if (cnt % 1000 == 0){
				writer.write(output.toString());
				writer.flush();
				output = new StringBuilder();
			}
		}
		writer.write(output.toString());
		writer.flush();
		output = new StringBuilder();
		writer.close();
		System.out.println("DONE");
		reader.close();
	}
}
