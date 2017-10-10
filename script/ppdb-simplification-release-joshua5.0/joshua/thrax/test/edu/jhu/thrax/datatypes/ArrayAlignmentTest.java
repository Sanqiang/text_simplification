package edu.jhu.thrax.datatypes;

import java.util.Iterator;

import org.testng.Assert;
import org.testng.annotations.Test;

public class ArrayAlignmentTest
{
	@Test
	public void sourceIndexIsAligned_IndexNotPresent_returnsFalse()
	{
		ArrayAlignment a = ArrayAlignment.fromString("0-0 2-2", false);
		Assert.assertFalse(a.sourceIndexIsAligned(1));
	}

	@Test
	public void targetIndexIsAligned_IndexNotPresent_returnsFalse() 
	{ 
		ArrayAlignment a = ArrayAlignment.fromString("0-0 2-2", false); 
		Assert.assertFalse(a.targetIndexIsAligned(1));
	}

	@Test
	public void sourceIndexIsAligned_IndexOutOfRange_returnsFalse()
	{
		ArrayAlignment a = ArrayAlignment.fromString("0-0 2-2", false); 
		Assert.assertFalse(a.sourceIndexIsAligned(-1));
		Assert.assertFalse(a.sourceIndexIsAligned(3));
	}

	@Test
	public void targetIndexIsAligned_IndexOutOfRange_returnsFalse()
	{
		ArrayAlignment a = ArrayAlignment.fromString("0-0 2-2", false); 
		Assert.assertFalse(a.targetIndexIsAligned(-1));
		Assert.assertFalse(a.targetIndexIsAligned(3));
	}

	@Test
	public void sourceIndexIsAligned_IndexPresent_returnsTrue()
	{
		ArrayAlignment a = ArrayAlignment.fromString("0-0 1-2 77-32", false);
		Assert.assertTrue(a.sourceIndexIsAligned(0));
		Assert.assertTrue(a.sourceIndexIsAligned(1));
		Assert.assertTrue(a.sourceIndexIsAligned(77));
	}

	@Test
	public void targetIndexIsAligned_IndexPresent_returnsTrue()
	{
		ArrayAlignment a = ArrayAlignment.fromString("0-0 1-2 77-32", false);
		Assert.assertTrue(a.targetIndexIsAligned(0));
		Assert.assertTrue(a.targetIndexIsAligned(2));
		Assert.assertTrue(a.targetIndexIsAligned(32));
	}

	@Test
	public void numTargetWordsAlignedTo_Unaligned_isZero()
	{
		ArrayAlignment a = ArrayAlignment.fromString("0-0 1-1", false);
		Assert.assertEquals(a.numTargetWordsAlignedTo(5), 0);
	}

	@Test
	public void numTargetWordsAlignedTo_Singleton_isOne()
	{
		ArrayAlignment a = ArrayAlignment.fromString("0-0 1-1", false);
		Assert.assertEquals(a.numTargetWordsAlignedTo(1), 1);
	}

	@Test
	public void numSourceWordsAlignedTo_Unaligned_isZero()
	{
		ArrayAlignment a = ArrayAlignment.fromString("0-0 1-1", false);
		Assert.assertEquals(a.numSourceWordsAlignedTo(5), 0);
	}

	@Test
	public void numSourceWordsAlignedTo_Singleton_isOne()
	{
		ArrayAlignment a = ArrayAlignment.fromString("0-0 1-1", false);
		Assert.assertEquals(a.numSourceWordsAlignedTo(1), 1);
	}

	@Test
	public void numTargetWordsAlignedTo_Multiple()
	{
		ArrayAlignment a = ArrayAlignment.fromString("1-3 1-1 1-77", false);
		Assert.assertEquals(a.numTargetWordsAlignedTo(1), 3);
	}

	@Test
	public void numSourceWordsAlignedTo_Multiple()
	{
		ArrayAlignment a = ArrayAlignment.fromString("1-3 1-1 1-77", true);
		Assert.assertEquals(a.numSourceWordsAlignedTo(1), 3);
	}

	@Test
	public void targetIndicesAlignedTo_Unaligned_EmptyIterator()
	{
		ArrayAlignment a = ArrayAlignment.fromString("1-3 5-1 6-77", false);
		Assert.assertFalse(a.targetIndicesAlignedTo(3).hasNext());
	}

	@Test
	public void sourceIndicesAlignedTo_Unaligned_EmptyIterator()
	{
		ArrayAlignment a = ArrayAlignment.fromString("1-3 5-1 6-77", false);
		Assert.assertFalse(a.sourceIndicesAlignedTo(15).hasNext());
	}

	@Test
	public void targetIndicesAlignedTo_Aligned_IteratorCorrect()
	{
		ArrayAlignment a = ArrayAlignment.fromString("1-3 5-77 5-1", false);
		Iterator<Integer> targetIndices = a.targetIndicesAlignedTo(5);
		int j;
		Assert.assertTrue(targetIndices.hasNext());
		j = targetIndices.next();
		Assert.assertEquals(j, 1);
		Assert.assertTrue(targetIndices.hasNext());
		j = targetIndices.next();
		Assert.assertEquals(j, 77);
		Assert.assertFalse(targetIndices.hasNext());
	}

	@Test
	public void sourceIndicesAlignedTo_Aligned_IteratorCorrect()
	{
		ArrayAlignment a = ArrayAlignment.fromString("1-3 5-77 5-3", false);
		Iterator<Integer> sourceIndices = a.sourceIndicesAlignedTo(3);
		int j;
		Assert.assertTrue(sourceIndices.hasNext());
		j = sourceIndices.next();
		Assert.assertEquals(j, 1);
		Assert.assertTrue(sourceIndices.hasNext());
		j = sourceIndices.next();
		Assert.assertEquals(j, 5);
		Assert.assertFalse(sourceIndices.hasNext());
	}


}

