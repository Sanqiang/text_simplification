package parser;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

import edu.stanford.nlp.ie.AbstractSequenceClassifier;
import edu.stanford.nlp.ie.crf.CRFClassifier;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreLabel;

public class NERParser {

	public static void main(String[] args) throws Exception {
		AbstractSequenceClassifier<CoreLabel> classifier1 = CRFClassifier.getClassifier(
				"/Users/zhaosanqiang916/git/stanford-ner-2017-06-09/classifiers/english.muc.7class.distsim.crf.ser.gz");
		AbstractSequenceClassifier<CoreLabel> classifier2 = CRFClassifier.getClassifier(
				"/Users/zhaosanqiang916/git/stanford-ner-2017-06-09/classifiers/english.conll.4class.distsim.crf.ser.gz");

		ArrayList<String> nlines = new ArrayList<>();
		BufferedReader reader = new BufferedReader(new FileReader(new File("/Volumes/Storage/wiki/text_simp.txt")));
		String line = null;
		int processed_idx = 0;
		while (null != (line = reader.readLine())) {
			List<List<CoreLabel>> llcl1 = classifier1.classify(line);
			List<List<CoreLabel>> llcl2 = classifier2.classify(line);
			assert llcl1.size() == llcl2.size();

			StringBuilder sbBuilder = new StringBuilder();
			for (int i = 0; i < llcl1.size(); i++) {
				List<CoreLabel> lcl1 = llcl1.get(i);
				List<CoreLabel> lcl2 = llcl2.get(i);
				assert lcl1.size() == lcl2.size();

				for (int j = 0; j < lcl1.size(); j++) {
					CoreLabel cl1 = lcl1.get(j);
					CoreLabel cl2 = lcl2.get(j);
					assert cl1.word().equals(cl2.word());

					String word = cl1.word();
					String ner1 = cl1.get(CoreAnnotations.AnswerAnnotation.class);
					String ner2 = cl2.get(CoreAnnotations.AnswerAnnotation.class);

					if (ner1.equals("O") && ner2.equals("O")) {
						sbBuilder.append(word);
					} else {
						sbBuilder.append(word).append("_TAG");
					}
					sbBuilder.append(" ");
				}
			}
			nlines.add(sbBuilder.toString());
			if (++processed_idx % 10000 == 0) {
				System.out.println("Processed" + processed_idx);
			}
		}
		reader.close();

		BufferedWriter writer = new BufferedWriter(
				new FileWriter(new File("/Volumes/Storage/wiki/text_simp_nertag.txt")));
		for (String nline : nlines) {
			writer.write(nline);
			writer.write("\n");
		}
		writer.close();
	}
}
