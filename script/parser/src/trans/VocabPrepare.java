package trans;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;

public class VocabPrepare {
	<K, V extends Comparable<? super V>> SortedSet<Map.Entry<K, V>> entriesSortedByValues(Map<K, V> map) {
		SortedSet<Map.Entry<K, V>> sortedEntries = new TreeSet<Map.Entry<K, V>>(new Comparator<Map.Entry<K, V>>() {
			@Override
			public int compare(Map.Entry<K, V> e1, Map.Entry<K, V> e2) {
				int res = e2.getValue().compareTo(e1.getValue());
				return res != 0 ? res : 1;
			}
		});
		sortedEntries.addAll(map.entrySet());
		return sortedEntries;
	}

	private HashMap<String, Integer> getMapDict(File file, HashMap<String, Integer> voc) throws Exception {
		BufferedReader reader = new BufferedReader(new FileReader(file));
		while (true) {
			String line = reader.readLine();
			if (line == null) {
				break;
			}
			String[] words = line.split(" ");
			for (String word : words) {
				word = word.toLowerCase();
				if (!voc.containsKey(word)) {
					voc.put(word, 0);
				}
				voc.put(word, 1 + voc.get(word));
			}
		}
		reader.close();
		return voc;
	}

	public void process(String path, String npath) throws Exception {
		File folder = new File(path);
		final File[] files = folder.listFiles();

		HashMap<String, Integer> voc_simp = new HashMap<>(), voc = new HashMap<>(), voc_comp = new HashMap<>();
		for (File file : files) {
			String name = file.getName();
			if (name.startsWith("ner_simp_")) {
				voc_simp = getMapDict(file, voc_simp);
				voc = getMapDict(file, voc);
				System.out.println("Finish " + file.getName());
			}
		}

		for (File file : files) {
			String name = file.getName();
			if (name.startsWith("ner_comp_")) {
				voc_comp = getMapDict(file, voc_comp);
				voc = getMapDict(file, voc);
				System.out.println("Finish " + file.getName());
			}
		}
		
		SortedSet<Map.Entry<String, Integer>> voc2 = entriesSortedByValues(voc);
		SortedSet<Map.Entry<String, Integer>> voc_comp2 = entriesSortedByValues(voc_comp);
		SortedSet<Map.Entry<String, Integer>> voc_simp2 = entriesSortedByValues(voc_simp);
		
		StringBuilder sb_all = new StringBuilder();
		for (Map.Entry<String, Integer> entry : voc2) {
			sb_all.append(entry.getKey()).append("\t").append(entry.getValue()).append("\n");
		}
		BufferedWriter writer_all = new BufferedWriter(new FileWriter(npath + "/voc_all.txt"));
		writer_all.write(sb_all.toString());
		writer_all.close();

		StringBuilder sb_comp = new StringBuilder();
		for (Map.Entry<String, Integer> entry : voc_comp2) {
			sb_comp.append(entry.getKey()).append("\t").append(entry.getValue()).append("\n");
		}
		BufferedWriter writer_comp = new BufferedWriter(new FileWriter(npath + "/voc_comp.txt"));
		writer_comp.write(sb_comp.toString());
		writer_comp.close();

		StringBuilder sb_simp = new StringBuilder();
		for (Map.Entry<String, Integer> entry : voc_simp2) {
			sb_simp.append(entry.getKey()).append("\t").append(entry.getValue()).append("\n");
		}
		BufferedWriter writer_simp = new BufferedWriter(new FileWriter(npath + "/voc_simp.txt"));
		writer_simp.write(sb_simp.toString());
		writer_simp.close();
	}

	public static void main(String[] args) throws Exception {
		new VocabPrepare().process("C:\\git\\wiki_output\\ner2", "C:\\git\\wiki_output\\voc");
	}
}
