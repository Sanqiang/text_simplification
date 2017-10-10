package edu.jhu.thrax.extraction;

import org.testng.Assert;
import org.testng.annotations.Test;

import edu.jhu.thrax.util.Vocabulary;

public class SAMTLabelerTest {
  
  private final int defaultLabel = Vocabulary.id("X");
  
  @Test
  public void getLabel_MalformedTree_isDefault() {
    SAMTLabeler labeler =
        new SAMTLabeler("(A b))", true, true, true, true, "top", defaultLabel);
    Assert.assertEquals(labeler.getLabel(0, 1), defaultLabel);
  }

  @Test
  public void getLabel_SpanOutOfBounds_isDefault() {
    SAMTLabeler labeler = new SAMTLabeler("(A b)", true, true, true, true, "top", defaultLabel);
    Assert.assertEquals(labeler.getLabel(0, 3), defaultLabel);
    Assert.assertEquals(labeler.getLabel(-2, 1), defaultLabel);
  }

  @Test
  public void getLabel_UnaryChain_Top() {
    SAMTLabeler labeler = new SAMTLabeler("(A (B c))", true, true, true, true, "top", defaultLabel);
    Assert.assertEquals(labeler.getLabel(0, 1), "A");
  }

  @Test
  public void getLabel_UnaryChain_Bottom() {
    SAMTLabeler labeler = new SAMTLabeler("(A (B c))", true, true, true, true, "bottom", defaultLabel);
    Assert.assertEquals(labeler.getLabel(0, 1), "B");
  }

  @Test
  public void getLabel_UnaryChain_All() {
    SAMTLabeler labeler = new SAMTLabeler("(A (B c))", true, true, true, true, "all", defaultLabel);
    Assert.assertEquals(labeler.getLabel(0, 1), "A:B");
  }

  @Test
  public void getLabel_NoConst_returnCat() {
    SAMTLabeler labeler = new SAMTLabeler("(A (B c) (D e))", false, true, true, true, "all", defaultLabel);
    Assert.assertEquals(labeler.getLabel(0, 2), "B+D");
  }

  @Test
  public void getLabel_NoConstCat_noCCG_returnDefault() {
    SAMTLabeler labeler = new SAMTLabeler("(A (B c) (D e))", false, true, false, true, "all", defaultLabel);
    Assert.assertEquals(labeler.getLabel(0, 2), defaultLabel);
  }

  @Test
  public void getLabel_NoConstCat_returnCCG() {
    SAMTLabeler labeler = new SAMTLabeler("(A (B c) (D e))", false, true, false, true, "all", defaultLabel);
    Assert.assertEquals(labeler.getLabel(0, 1), "A/D");
    Assert.assertEquals(labeler.getLabel(1, 2), "A\\B");
  }

  @Test
  public void getLabel_NoConstCatCCG_returnDoubleCat() {
    SAMTLabeler labeler =
        new SAMTLabeler("(A (B c) (D e) (F g))", false, false, false, true, "all", defaultLabel);
    Assert.assertEquals(labeler.getLabel(0, 3), "B+D+F");
  }
}
