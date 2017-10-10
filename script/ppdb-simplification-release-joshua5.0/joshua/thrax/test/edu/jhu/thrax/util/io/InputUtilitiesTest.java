package edu.jhu.thrax.util.io;

import org.testng.Assert;
import org.testng.annotations.Test;

import edu.jhu.thrax.datatypes.AlignedSentencePair;
import edu.jhu.thrax.util.exceptions.MalformedInputException;

public class InputUtilitiesTest
{
    @Test
    public void parseYield_EmptyString_ReturnsZeroLengthArray() throws MalformedInputException
    {
        Assert.assertEquals(InputUtilities.parseYield("").length, 0);
    }

    @Test
    public void parseYield_Whitespace_ReturnsZeroLengthArray() throws MalformedInputException
    {
        Assert.assertEquals(InputUtilities.parseYield("        ").length, 0);
    }
    @Test
    public void parseYield_EmptyParse_ReturnsZeroLengthArray() throws MalformedInputException
    {
        Assert.assertEquals(InputUtilities.parseYield("()").length, 0);
    }

    @Test(expectedExceptions = { MalformedInputException.class })
    public void parseYield_UnbalancedLeft_ThrowsException() throws MalformedInputException
    {
        InputUtilities.parseYield("(S (DT the) (NP dog)");
    }

    @Test(expectedExceptions = { MalformedInputException.class })
    public void parseYield_UnbalancedRight_ThrowsException() throws MalformedInputException
    {
        InputUtilities.parseYield("(S (DT the) (NP dog)))");
    }

    @Test
    public void getWords_EmptyString_ReturnsZeroLengthArray() throws MalformedInputException
    {
        Assert.assertEquals(InputUtilities.getWords("", false).length, 0);
        Assert.assertEquals(InputUtilities.getWords("", true).length, 0);
    }

    @Test
    public void getWords_Whitespace_ReturnsZeroLengthArray() throws MalformedInputException
    {
        Assert.assertEquals(InputUtilities.getWords("    ", false).length, 0);
        Assert.assertEquals(InputUtilities.getWords("    ", true).length, 0);
    }

	@Test
    public void getWords_PlainWords_ReturnsStringArray() throws MalformedInputException
    {
        String [] tokens = { "hello", ",", "world" };
        Assert.assertEquals(InputUtilities.getWords("hello , world", false), tokens);
    }

    @Test(expectedExceptions = { MalformedInputException.class })
	public void alignedSentencePair_emptyString_throwsException() throws MalformedInputException
	{
		InputUtilities.alignedSentencePair("", false, false, false);
	}

    @Test(expectedExceptions = { MalformedInputException.class })
	public void alignedSentencePair_twoFields_throwsException() throws MalformedInputException
	{
		InputUtilities.alignedSentencePair("foo ||| bar", false, false, false);
	}

    @Test(expectedExceptions = { MalformedInputException.class })
	public void alignedSentencePair_badAlignment_throwsException() throws MalformedInputException
	{
		InputUtilities.alignedSentencePair("foo ||| bar ||| 0-1", false, false, false);
	}

    @Test(expectedExceptions = { MalformedInputException.class })
	public void alignedSentencePair_emptyField_throwsException() throws MalformedInputException
	{
		InputUtilities.alignedSentencePair("foo ||| ||| 0-0", false, false, false);
	}

    @Test
	public void alignedSentencePair_simple() throws MalformedInputException
	{
		AlignedSentencePair pair = InputUtilities.alignedSentencePair("foo ||| bar ||| 0-0", false, false, false);
		Assert.assertEquals(pair.source.length, 1);
		Assert.assertEquals(pair.source[0], "foo");
		Assert.assertEquals(pair.target.length, 1);
		Assert.assertEquals(pair.target[0], "bar");
		Assert.assertEquals(pair.alignment.numTargetWordsAlignedTo(0), 1);
		Assert.assertEquals(pair.alignment.numSourceWordsAlignedTo(0), 1);
	}

    @Test
	public void alignedSentencePair_reversed() throws MalformedInputException
	{
		AlignedSentencePair pair = InputUtilities.alignedSentencePair("foo ||| bar ||| 0-0", false, false, true);
		Assert.assertEquals(pair.source.length, 1);
		Assert.assertEquals(pair.source[0], "bar");
		Assert.assertEquals(pair.target.length, 1);
		Assert.assertEquals(pair.target[0], "foo");
		Assert.assertEquals(pair.alignment.numTargetWordsAlignedTo(0), 1);
		Assert.assertEquals(pair.alignment.numSourceWordsAlignedTo(0), 1);
	}
}

