package edu.jhu.thrax.syntax;

import java.util.Iterator;
import java.util.List;

import org.testng.Assert;
import org.testng.annotations.Test;

public class ParseTreeTest
{
	@Test
	public void numLeaves_Leaf_isOne()
	{
		ParseTree pt = ParseTree.fromPennFormat("a");
		Assert.assertEquals(pt.numLeaves(), 1);
	}

	@Test
	public void numNodes_Leaf_isOne()
	{
		ParseTree pt = ParseTree.fromPennFormat("a");
		Assert.assertEquals(pt.numNodes(), 1);
	}

	@Test
	public void numLeaves_Tree()
	{
		ParseTree pt = ParseTree.fromPennFormat("(A (B c d))");
		Assert.assertEquals(pt.numLeaves(), 2);
	}

	@Test
	public void numNodes_Tree()
	{
		ParseTree pt = ParseTree.fromPennFormat("(A (B c d))");
		Assert.assertEquals(pt.numNodes(), 4);
	}

	@Test 
	void internalNodesWithSpan_Single()
	{
		ParseTree pt = ParseTree.fromPennFormat("(A (B c d) e)");
		List<ParseTree.Node> list = pt.internalNodesWithSpan(0, 2);
		Assert.assertEquals(list.size(), 1);
		ParseTree.Node node = list.get(0);
		Assert.assertEquals(node.label(), "B");
		Assert.assertEquals(node.spanStart(), 0);
		Assert.assertEquals(node.spanEnd(), 2);
		Assert.assertFalse(node.numChildren() == 0);
	}

	@Test
	public void internalNodesWithSpan_unaryChain()
	{
		ParseTree pt = ParseTree.fromPennFormat("(A (B c))");
		List<ParseTree.Node> list = pt.internalNodesWithSpan(0, 1);
		Assert.assertEquals(list.size(), 2);
		ParseTree.Node node = list.get(0);
		Assert.assertEquals(node.label(), "A");
		Assert.assertEquals(node.spanStart(), 0);
		Assert.assertEquals(node.spanEnd(), 1);
		Assert.assertFalse(node.numChildren() == 0);
		node = list.get(1);
		Assert.assertEquals(node.label(), "B");
		Assert.assertEquals(node.spanStart(), 0);
		Assert.assertEquals(node.spanEnd(), 1);
		Assert.assertFalse(node.numChildren() == 0);
	}

	@Test
	public void leaf_ChildIterator_isEmpty()
	{
		ParseTree pt = ParseTree.fromPennFormat("a");
		ParseTree.Node node = pt.root();
		Assert.assertFalse(node.children().hasNext());
	}

	@Test
	public void tree_ChildIterator()
	{
		ParseTree pt = ParseTree.fromPennFormat("(A b c)");
		ParseTree.Node node = pt.root();
		Iterator<ParseTree.Node> iter = node.children();
		Assert.assertTrue(iter.hasNext());
		node = iter.next();
		Assert.assertEquals(node.label(), "b");
		Assert.assertEquals(node.spanStart(), 0);
		Assert.assertEquals(node.spanEnd(), 1);
		Assert.assertFalse(node.children().hasNext());
		Assert.assertTrue(iter.hasNext());
		node = iter.next();
		Assert.assertEquals(node.label(), "c");
		Assert.assertEquals(node.spanStart(), 1);
		Assert.assertEquals(node.spanEnd(), 2);
		Assert.assertFalse(node.children().hasNext());
		Assert.assertFalse(iter.hasNext());
	}
}

