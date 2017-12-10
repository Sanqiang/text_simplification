package trans;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.OutputStreamWriter;

import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.chrome.ChromeDriver;

public class TransPrepare {

	private final int BATCH_SIZE = 16;
	WebDriver driver = null;

	public TransPrepare() {
		System.setProperty("webdriver.chrome.driver", "C:\\git\\java\\selenium-java-3.8.1\\chromedriver.exe");
		driver = new ChromeDriver();
		driver.get("https://translate.google.com/?sl=en&tl=zh-CN");
	}

	public String getChineseText(String text) {
		driver.findElement(By.id("source")).clear();
		driver.findElement(By.id("source")).sendKeys(text);
		driver.findElement(By.id("gt-submit")).click();

		String result = driver.findElement(By.id("gt-res-dir-ctr")).getText();
		// System.out.println(result);
		// driver.close();

		return result;
	}

	public void translate(String path, String npath) {
		try {
			BufferedWriter writer =new BufferedWriter(  
			        new OutputStreamWriter(  
			                new FileOutputStream(npath), "UTF-8"));
			BufferedReader reader = new BufferedReader(new FileReader(new File(path)));
			String line = null;
			StringBuilder text = new StringBuilder(), ntext = new StringBuilder();
			int line_id = 0;
			while (null != (line = reader.readLine())) {
				text.append(line);
				text.append("\n");

				++line_id;
				if (line_id % BATCH_SIZE == 0) {
					ntext.append(getChineseText(text.toString()));
					text = new StringBuilder();
				}
				if (ntext.length() >= 100000) {
					writer.write(ntext.toString());
					writer.flush();
					ntext = new StringBuilder();
					System.out.println("Processed:" + line_id);
				}
			}
			reader.close();
			writer.write(ntext.toString());
			writer.close();
		} catch (Exception e) {
			System.out.println(e.getMessage());
		}

	}

	public static void main(String[] args) {
		TransPrepare prepare = new TransPrepare();
		prepare.translate("comp_all.txt", "comp_all2.txt");
	}
}
