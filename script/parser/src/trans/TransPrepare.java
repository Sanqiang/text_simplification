package trans;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ForkJoinPool;

import org.openqa.selenium.By;
import org.openqa.selenium.JavascriptExecutor;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import com.google.common.base.CharMatcher;

public class TransPrepare {

	public TransPrepare() {
		System.setProperty("webdriver.chrome.driver", "C:\\git\\java\\selenium-java-3.8.1\\chromedriver.exe");
		// System.setProperty("webdriver.chrome.driver",
		// "/Users/zhaosanqiang916/git/selenium-java-3.8.1/chromedriver");

		// System.setProperty("webdriver.gecko.driver",
		// "/Users/zhaosanqiang916/git/selenium-java-3.8.1/geckodriver");

	}

	private String getChineseText(String text, boolean back, WebDriver driver_tmp) throws Exception {
		if (back) {
			driver_tmp.get("https://translate.google.com/?hl=en&sl=zh-CN&tl=en");
		} else {
			driver_tmp.get("https://translate.google.com/?hl=en&sl=en&tl=zh-CN");
		}
		driver_tmp.findElement(By.id("source")).clear();
		WebDriverWait wait0 = new WebDriverWait(driver_tmp, 60);
		wait0.until(ExpectedConditions.textToBe(By.id("result_box"), ""));

		// driver_tmp.findElement(By.id("source")).sendKeys(text);
		((JavascriptExecutor) driver_tmp).executeScript("arguments[0].value = arguments[1];",
				driver_tmp.findElement(By.id("source")), text);

		driver_tmp.findElement(By.id("gt-submit")).click();
		WebDriverWait wait = new WebDriverWait(driver_tmp, 60);
		if (back) {
			wait.until(ExpectedConditions.attributeToBe(driver_tmp.findElement(By.id("result_box")), "lang", "en"));
		} else {
			wait.until(ExpectedConditions.attributeToBe(driver_tmp.findElement(By.id("result_box")), "lang", "zh-CN"));
		}
		String result = driver_tmp.findElement(By.id("gt-res-dir-ctr")).getText();
		return result;
	}

	public String getChineseTextByFile(String path, boolean back) throws Exception {
		WebDriver driver_tmp = new ChromeDriver();
		try {
			if (back) {
				driver_tmp.get("https://translate.google.com/?tr=f&hl=en&sl=zh-CN&tl=en");
			} else {
				driver_tmp.get("https://translate.google.com/?tr=f&hl=en&sl=en&tl=zh-CN");
			}
			driver_tmp.findElement(By.id("file")).sendKeys(path);
			driver_tmp.findElement(By.id("gt-submit")).click();

			WebDriverWait wait = new WebDriverWait(driver_tmp, 60);
			wait.until(ExpectedConditions.urlToBe("https://translate.googleusercontent.com/translate_f"));
			WebElement content = driver_tmp.findElement(By.tagName("body"));
			String result = content.getText();
			driver_tmp.close();
			return result;
		} catch (Exception e) {
			driver_tmp.close();
			return null;
		}

	}

	private void trans1(File file) throws Exception {
		// Translate a single file from English to Chinese
		// Used for translateByFile as multithread unit
		// if (!file.getName().startsWith("simple")) {
		// continue;
		// }
		String npath = file.getParent() + "/tran1/" + file.getName();
		if (new File(npath).exists()) {
			return;
		}
		String result = getChineseTextByFile(file.getAbsolutePath(), false);
		if (result == null) {
			return;
		}
		BufferedWriter writer = new BufferedWriter(new FileWriter(new File(npath)));
		writer.write(result);
		writer.close();
	}

	private void trans2(File file) throws Exception {
		// Translate a single file from Chinese to English
		// Used for translateByFile as multithread unit
		// if (!(file.getName().startsWith("simple") &&
		// file.getName().endsWith(".t.txt"))) {
		// continue;
		// }
		String npath = file.getParent() + "/tran2/" + file.getName();
		if (new File(npath).exists()) {
			return;
		}
		String result = getChineseTextByFile(file.getAbsolutePath(), true);
		if (result == null) {
			return;
		}
		BufferedWriter writer = new BufferedWriter(new FileWriter(new File(npath)));
		writer.write(result);
		writer.close();
	}

	public void translateByFile(String path) throws Exception {
		ForkJoinPool pool = new ForkJoinPool(6);
		File folder = new File(path);
		final File[] files = folder.listFiles();
		pool.submit(() -> Arrays.asList(files).parallelStream().forEach(f -> {
			try {
				trans1(f);
			} catch (Exception e) {
				e.printStackTrace();
			}
		})).get();

		// for (File file : files) {
		// trans1(file);
		// }

		final File[] files2 = new File("/Volumes/Storage/wiki/wiki_output/comp/tran1").listFiles();
		pool.submit(() -> Arrays.asList(files2).parallelStream().forEach(f -> {
			try {
				trans2(f);
			} catch (Exception e) {
				e.printStackTrace();
			}
		})).get();
		// for (File file : files) {
		// tran2(file);
		// }
	}

	ConcurrentHashMap<Long, WebDriver> driver_pool = new ConcurrentHashMap<>();

	private File trans3(File file, boolean back) throws Exception {
		// translate single file chunk by chunk
		// used for translate
		String npath = null;
		if (back) {
			npath = file.getParent() + "/tran2/" + file.getName();
		} else {
			npath = file.getParent() + "/tran1/" + file.getName();
		}
		// System.out.println("Start:\t" + npath + "\t with back:\t" + back);
		File nfile = new File(npath);
		if (nfile.exists()) {
			// System.out.println("Ignore:\t" + npath);
			return nfile;
		}
		long threadId = Thread.currentThread().getId();
		if (!driver_pool.containsKey(threadId)) {
			System.out.println("Create driver for id:\t" + threadId);
			driver_pool.put(threadId, new ChromeDriver());
		} else {
			System.out.println("Get driver for id:\t" + threadId);
		}
		WebDriver driver_tmp = driver_pool.get(threadId);
		BufferedReader reader = new BufferedReader(new FileReader(file));
		try {
			StringBuilder output = new StringBuilder();
			StringBuilder text = new StringBuilder();
			String line = null;
			while (null != (line = reader.readLine())) {
				if (text.toString().length() + line.toString().length() + "\nzzzzz\n".length() >= 3000) {
					String ntext = getChineseText(text.toString(), back, driver_tmp);
					output.append(ntext.replaceAll("ZZZZZ", "\n").replaceAll("\n\n", "\n").replaceAll("\n\n", "\n"));
					text = new StringBuilder();
				}
				text.append(line.toString());
				if (back) {
					text.append("\nZZZZZ\n");
				} else {
					text.append("\nzzzzz\n");
				}
			}

			BufferedWriter writer = new BufferedWriter(new FileWriter(nfile));
			String ntext = getChineseText(text.toString(), back, driver_tmp);
			output.append(ntext.replaceAll("ZZZZZ", "\n").replaceAll("\n\n", "\n").replaceAll("\n\n", "\n"));
			writer.write(output.toString());
			writer.close();

		} catch (Exception e) {
			// driver_tmp.quit();
			// driver_tmp.close();
			reader.close();

			return null;
		}
		// driver_tmp.quit();
		// driver_tmp.close();
		reader.close();
		// System.out.println("Generate:\t" + npath);
		return nfile;
	}

	public void translate(String path) throws Exception {
		int num_thread = 30;
		ForkJoinPool pool = new ForkJoinPool(num_thread);

		File folder = new File(path);
		final File[] files = folder.listFiles();

		pool.submit(() -> Arrays.asList(files).parallelStream().forEach(f -> {
			File nfile = null;
			try {
				nfile = trans3(f, false);
				if (nfile != null) {
					trans3(nfile, true);
				}
			} catch (Exception e) {
				e.printStackTrace();
			} finally {
				if (nfile != null) {
					nfile = null;
				}
			}
		})).get();

		// try {
		//
		// BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new
		// FileOutputStream(npath), "UTF-8"));
		// BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
		// String line = null;
		// StringBuilder text = new StringBuilder(), ntext = new StringBuilder();
		// int line_id = 0;
		// while (null != (line = reader.readLine())) {
		// text.append(line);
		// text.append("\n");
		//
		// ++line_id;
		// if (line_id % BATCH_SIZE == 0) {
		// //ntext.append(getChineseText(text.toString()));
		// text = new StringBuilder();
		// }
		// if (ntext.length() >= 100000) {
		// writer.write(ntext.toString());
		// writer.flush();
		// ntext = new StringBuilder();
		// System.out.println("Processed:" + line_id);
		// }
		// }
		// reader.close();
		// writer.write(ntext.toString());
		// writer.close();
		// } catch (Exception e) {
		// System.out.println(e.getMessage());
		// }
	}

	private String text_postprocess(String text) {
		text = text.replaceAll("[- ]*L[A-Z]?B+[- ]*", " -LRB- ");
		text = text.replaceAll("[- ]*[^L]R[A-Z]?B+[- ]*", " -RRB- ");
		text = text.replace(" - ", "");
		text = text.replace(" +", " ");
		return text;
	}

	public void postprocess(String comp_path, String simp_path, String ncomp_path, String nsimp_path, int num_part)
			throws Exception {

		ArrayList<BufferedWriter> writer_comps = new ArrayList<>(), writer_simps = new ArrayList<>();
		;
		for (int i = 0; i < num_part; i++) {
			writer_comps.add(new BufferedWriter(new FileWriter(ncomp_path + i + ".txt")));
			writer_simps.add(new BufferedWriter(new FileWriter(nsimp_path + i + ".txt")));
		}
		int part_idx = 0;

		ArrayList<String> ncomp_sents = new ArrayList<>();
		ArrayList<String> nsimp_sents = new ArrayList<>();

		File folder = new File(simp_path);
		final File[] files = folder.listFiles();
		for (File simp_file : files) {
			String file_name = simp_file.getName();
			File comp_file = new File(comp_path + "/" + file_name);
			if (!comp_file.exists()) {
				continue;
			}

			BufferedReader reader_comp = new BufferedReader(new FileReader(comp_file));
			int lines_comp = 0;
			while (reader_comp.readLine() != null)
				lines_comp++;
			reader_comp.close();

			BufferedReader reader_simp = new BufferedReader(new FileReader(simp_file));
			int lines_simp = 0;
			while (reader_simp.readLine() != null)
				lines_simp++;
			reader_simp.close();

			if (lines_comp == lines_simp) {
				reader_comp = new BufferedReader(new FileReader(comp_file));
				reader_simp = new BufferedReader(new FileReader(simp_file));
				while (true) {
					String line_comp = reader_comp.readLine();
					String line_simp = reader_simp.readLine();
					if (line_comp == null || line_simp == null) {
						break;
					}
					String[] line_comp_words = line_comp.split(" ");
					String[] line_simp_words = line_simp.split(" ");
					// Checker: length >= 10
					if (line_comp_words.length < 10 || line_simp_words.length < 10) {
						continue;
					}
					// Checker: all asc2 code
					boolean validate = true;
					for (String word : line_simp_words) {
						if (!CharMatcher.ascii().matchesAllOf(word)) {
							validate = false;
							break;
						}
					}
					if (!validate) {
						continue;
					}
					for (String word : line_comp_words) {
						if (!CharMatcher.ascii().matchesAllOf(word)) {
							validate = false;
							break;
						}
					}
					if (!validate) {
						continue;
					}

					ncomp_sents.add(line_comp);
					nsimp_sents.add(line_simp);
					if (ncomp_sents.size() > 100000) {
						StringBuilder sb_comp = new StringBuilder();
						for (String tmp : ncomp_sents) {
							sb_comp.append(tmp).append("\n");
						}
						int cur_part_idx = part_idx++ % num_part;
						writer_comps.get(cur_part_idx).write(text_postprocess(sb_comp.toString()));
						writer_comps.get(cur_part_idx).flush();
						ncomp_sents.clear();

						StringBuilder sb_simp = new StringBuilder();
						for (String tmp : nsimp_sents) {
							sb_simp.append(tmp).append("\n");
						}
						writer_simps.get(cur_part_idx).write(text_postprocess(sb_simp.toString()));
						writer_simps.get(cur_part_idx).flush();
						nsimp_sents.clear();
					}
				}

				int cur_part_idx = part_idx++ % num_part;
				StringBuilder sb_comp = new StringBuilder();
				for (String tmp : ncomp_sents) {
					sb_comp.append(tmp).append("\n");
				}
				writer_comps.get(cur_part_idx).write(text_postprocess(sb_comp.toString()));
				writer_comps.get(cur_part_idx).flush();
				ncomp_sents.clear();

				StringBuilder sb_simp = new StringBuilder();
				for (String tmp : nsimp_sents) {
					sb_simp.append(tmp).append("\n");
				}
				writer_simps.get(cur_part_idx).write(text_postprocess(sb_simp.toString()));
				writer_simps.get(cur_part_idx).flush();
				nsimp_sents.clear();
			} else {
				System.out.println(file_name);
			}
		}
	}

	public static void main(String[] args) throws Exception {
		TransPrepare prepare = new TransPrepare();
		prepare.postprocess("C:\\git\\wiki_output\\comp", "C:\\git\\wiki_output\\comp\\tran1\\tran2",
				"C:\\git\\wiki_output\\compf_", "C:\\git\\wiki_output\\simpf_", 7);
		// prepare.translate("C:\\git\\wiki_output\\comp");
		// prepare.translate("/Volumes/Storage/wiki/wiki_output/comp/");
		// prepare.translateByFile("/Volumes/Storage/wiki/wiki_output/comp/");
	}
}
