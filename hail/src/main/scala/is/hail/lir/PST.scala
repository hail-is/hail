package is.hail.lir

import is.hail.expr.ir.{BooleanArrayBuilder, IntArrayBuilder}
import is.hail.utils.BoxedArrayBuilder

import scala.collection.mutable

// PST computes a non-standard variant of the Program Structure Tree (PST)
// For the original definition, see:
// Johnson, Pearson, Pingali, The program structure tree: computing control regions in linear time
// https://dl.acm.org/doi/10.1145/773473.178258

// Our PST is defined as follows:

// A _region_ is a subset of nodes of the control flow graph
// (and hence a subgraph of the control flow graph)
// with two distinguished nodes (start, end) such that
// all edges from outside the region into the region target start
// and all edges from the region to the outside leave end.

// Regions are the units of code which will be split out into
// separate methods by SplitMethod.

// A region is canonical if it cannot be split into two subregions.
// A Program Structure Tree is the tree of canonical regions
// organized by containment.  Two regions are either disjoint or
// one is contained within the other.  The root of the PST is
// the entire function.

// The algorithm is organized as follows:
//  - Perform a depth first search to determine the back edges.
//    All other edges are forward edges.
//  - The forward edges form a DAG, linearize the DAG placing
//    children as close as possible to parents.
//  - Walking forward to find the regions.  This is done walking
//    forward along the nodes considering the forward edges, then
//    verifying candidate regions are regions in light of back edges.
//    This is done in a way that allows regions to overlap at start
//    and end.  (Consider the stacked double diamond CFG coming from
//    two sequential conditionals.)  This step does not include
//    singleton regions (so the regions found still form a tree.)
//  - Split shared blocks and reindex the blocks in linearized
//    order, adding singleton regions.

// A note on complexity. There probably a linear-time algorithm to compute
// this without linearization, but I found this conceptually simple to
// implement and it will be more than fast enough.  Most steps are linear,
// except:
//  - it is O(N log N) in the back edges (a minority of edges) because of
//    the use of TreeSet when computing minTargetLE and maxSourceGE,
// - and O(N^2) to identify seqential regions when checking regionStarts,
//   although that is only non-trivial when the CFG is irreducible, which
//   is relatively rare.

class PSTRegion(
  var start: Int,
  var end: Int,
  var children: Array[Int],
  // -1 means root, or parent not yet known
  var parent: Int = -1,
)

object PSTResult {
  def unapply(result: PSTResult): Option[(Blocks, CFG, PST)] =
    Some((result.blocks, result.cfg, result.pst))
}

class PSTResult(
  val blocks: Blocks,
  val cfg: CFG,
  val pst: PST,
)

class PSTBuilder(
  m: Method,
  blocks: Blocks,
  cfg: CFG,
) {
  def nBlocks: Int = blocks.length

  // for debugging
  private def checkRegion(start: Int, end: Int): Unit = {
    // only enter at start
    var i = start + 1
    while (i <= end) {
      val b = linearization(i)
      for (p <- cfg.pred(b)) {
        val j = blockLinearIdx(p)
        assert(j >= start && j <= end)
      }
      i += 1
    }

    // only exit end
    i = start
    while (i < end) {
      val b = linearization(i)
      for (s <- cfg.succ(b)) {
        val j = blockLinearIdx(s)
        assert(j >= start || j <= end)
      }
      i += 1
    }
  }

  private val backEdges = mutable.Set[(Int, Int)]()

  private def computeBackEdges(): Unit = {
    // recursion will blow out the stack
    val stack = new BoxedArrayBuilder[(Int, Iterator[Int])]()
    val onStack = mutable.Set[Int]()
    val visited = mutable.Set[Int]()

    def push(i: Int): Unit = {
      stack.push(i -> cfg.succ(i).iterator)
      onStack += i
      visited += i
    }

    def pop(): Unit = {
      val (i, _) = stack.pop()
      onStack -= i
    }

    push(cfg.entry)
    while (stack.nonEmpty) {
      val (i, it) = stack.top
      if (it.hasNext) {
        val s = it.next()
        if (onStack(s)) {
          backEdges += i -> s
        } else {
          if (!visited(s))
            push(s)
        }
      } else
        pop()
    }

    assert(onStack.isEmpty)
  }

  private val linearization = new Array[Int](nBlocks)

  // successors
  // in linear index
  private val forwardSucc = Array.fill(nBlocks)(mutable.Set[Int]())
  private val backwardSucc = Array.fill(nBlocks)(mutable.Set[Int]())
  // predecessors
  private val backwardPred = Array.fill(nBlocks)(mutable.Set[Int]())

  private val blockLinearIdx = new Array[Int](nBlocks)

  private def linearize(): Unit = {
    val pending = Array.tabulate(nBlocks) { i =>
      var n = 0
      for (p <- cfg.pred(i))
        if (!backEdges(p -> i))
          n += 1
      n
    }
    var k = 0

    // recursion will blow out the stack
    val stack = new BoxedArrayBuilder[(Int, Iterator[Int])]()

    def push(b: Int): Unit = {
      val i = k
      k += 1
      linearization(i) = b
      blockLinearIdx(b) = i
      stack.push(b -> cfg.succ(b).iterator)
    }

    push(cfg.entry)
    while (stack.nonEmpty) {
      val (b, it) = stack.top
      if (it.hasNext) {
        val s = it.next()
        if (!backEdges(b -> s)) {
          pending(s) -= 1
          if (pending(s) == 0) {
            push(s)
          }
        }
      } else
        stack.pop()
    }

    assert(k == nBlocks)

    var i = 0
    while (i < nBlocks) {
      val b = linearization(i)
      for (p <- cfg.succ(b)) {
        val j = blockLinearIdx(p)
        if (i < j) {
          assert(!backEdges(b -> p))
          forwardSucc(i) += j
        } else if (i > j) {
          assert(backEdges(b -> p))
          backwardSucc(i) += j
          backwardPred(j) += i
        }
      }
      i += 1
    }
  }

  // maxTargetLE(i) is the largest back edge target <= i
  // from a back edge with source greater than i
  private val maxTargetLE = new Array[Int](nBlocks)

  private def computeMaxTargetLE(): Unit = {
    var i = nBlocks - 1
    val targets = new java.util.TreeSet[java.lang.Integer]()
    while (i >= 0) {
      if (targets.isEmpty)
        maxTargetLE(i) = -1
      else
        maxTargetLE(i) = targets.last()
      targets.remove(i)
      for (s <- backwardSucc(i))
        targets.add(s)
      i -= 1
    }
  }

  // minSourceGE(i) is the smallest back edge source >= i
  // from a back edge with target smaller than i
  private val minSourceGE = new Array[Int](nBlocks)

  private def computeMinSourceGE(): Unit = {
    var i = 0
    val sources = new java.util.TreeSet[java.lang.Integer]()
    while (i < nBlocks) {
      if (sources.isEmpty)
        minSourceGE(i) = -1
      else
        minSourceGE(i) = sources.first()
      sources.remove(i)
      for (p <- backwardPred(i))
        sources.add(p)
      i += 1
    }
  }

  private def backEdgesOK(start: Int, end: Int): Boolean = {
    val maxTarget = maxTargetLE(end)
    val minSource = minSourceGE(start)
    (maxTarget == -1 || maxTarget <= start) &&
    (minSource == -1 || minSource >= end)
  }

  private val splitBlock = new java.util.BitSet(nBlocks)

  private val regions: mutable.ArrayBuffer[PSTRegion] = mutable.ArrayBuffer[PSTRegion]()

  // regions with no parents
  private val frontier = new IntArrayBuilder()

  private def addRegion(start: Int, end: Int): Int = {
    var firstc = frontier.size
    while ((firstc - 1) >= 0 && regions(frontier(firstc - 1)).start >= start)
      firstc -= 1
    assert(firstc == 0 || regions(frontier(firstc - 1)).end <= start)

    val ri = regions.length
    val n = frontier.size - firstc
    val children = new Array[Int](n)
    var i = 0
    while (i < n) {
      val c = frontier(firstc + i)
      assert(regions(c).parent == -1)
      regions(c).parent = ri
      children(i) = c
      i += 1
    }
    frontier.setSizeUninitialized(frontier.size - n)
    if (frontier.size > 0 && regions(frontier(frontier.size - 1)).end == start)
      splitBlock.set(start)
    frontier += ri
    regions += new PSTRegion(start, end, children)
    ri
  }

  private def addRoot(): Int = {
    if (
      frontier.size == 1 &&
      regions(frontier(0)).start == 0 &&
      regions(frontier(0)).end == nBlocks - 1
    ) {
      frontier(0)
    } else {
      val c = regions.length

      val ri = regions.length
      val n = frontier.size
      val children = new Array[Int](n)
      var i = 0
      while (i < n) {
        val c = frontier(i)
        assert(regions(c).parent == -1)
        regions(c).parent = ri
        children(i) = c
        i += 1
      }
      regions += new PSTRegion(0, nBlocks - 1, children)
      frontier.clear()
      frontier += c
      c
    }
  }

  // find regions in [start, end]
  // no edges from [0, start) target (start, end]
  private def findRegions(start: Int, end: Int): Unit = {
    var regionStarts = new IntArrayBuilder()
    regionStarts += start

    // find subregions of [start, end]

    // forward edge targets from [start, newStart) into [newStart, ...
    val targets = new java.util.TreeSet[java.lang.Integer]()
    for (s <- forwardSucc(start))
      targets.add(s)
    var newStart = start + 1

    while (newStart <= end) {
      targets.remove(newStart)

      if (targets.isEmpty) {
        // find smallest region with newStart as end block
        // if it exists
        @scala.annotation.tailrec
        def f(i: Int): Unit = {
          if (i >= 0) {
            val rStart = regionStarts(i)

            if (backEdgesOK(rStart, newStart)) {
              // expensive, for debugging
              checkRegion(rStart, newStart)
              addRegion(rStart, newStart)
            } else
              f(i - 1)
          }
        }

        f(regionStarts.size - 1)

        regionStarts += newStart
      }

      val nextStart: Int =
        if (targets.isEmpty)
          newStart + 1
        else
          targets.first()
      assert(nextStart > newStart)

      var newEnd = nextStart - 1
      if (newEnd > end)
        newEnd = end

      findRegions(newStart, newEnd)

      var i = newStart
      while (i <= newEnd) {
        targets.remove(i)
        for (s <- forwardSucc(i))
          targets.add(s)
        i += 1
      }

      newStart = newEnd + 1
    }
  }

  def result(): PSTResult = {
    computeBackEdges()
    linearize()
    computeMaxTargetLE()
    computeMinSourceGE()
    findRegions(0, nBlocks - 1)
    val root = addRoot()

    val newBlocksB = new BoxedArrayBuilder[Block]()
    val newSplitBlock = new BooleanArrayBuilder()

    // split blocks, compute new blocks
    // in linearization order
    val blockNewEndBlockIdx = new Array[Int](nBlocks)
    val blockNewStartBlockIdx = new Array[Int](nBlocks)
    var i = 0
    while (i < nBlocks) {
      val b = blocks(linearization(i))
      if (splitBlock.get(i)) {
        val splitb = new Block()
        splitb.method = m
        val last = b.last
        last.remove()
        splitb.append(last)
        b.append(goto(splitb))
        val newi = newBlocksB.length
        newBlocksB += b
        newSplitBlock += true
        newBlocksB += splitb
        newSplitBlock += false
        blockNewEndBlockIdx(i) = newi
        blockNewStartBlockIdx(i) = newi + 1
      } else {
        val newi = newBlocksB.length
        newBlocksB += b
        newSplitBlock += false
        blockNewEndBlockIdx(i) = newi
        blockNewStartBlockIdx(i) = newi
      }
      i += 1
    }

    // update start, end
    i = 0
    while (i < regions.length) {
      val r = regions(i)
      r.start = blockNewStartBlockIdx(r.start)
      r.end = blockNewEndBlockIdx(r.end)
      i += 1
    }

    // compute new regions, including singletons
    // update children
    val newRegionsB = new BoxedArrayBuilder[PSTRegion]()
    val regionNewRegion = new Array[Int](regions.length)
    i = 0
    while (i < regions.length) {
      val r = regions(i)
      val children = r.children

      var c = 0
      var ci = 0
      var child: PSTRegion = null
      if (c < children.length) {
        ci = children(c)
        assert(ci < i)
        child = regions(children(c))
      }

      val newChildren = new IntArrayBuilder()

      var j = r.start
      var jincluded = false
      while (j <= r.end) {
        if (child != null && child.start == j) {
          newChildren += regionNewRegion(ci)
          j = child.end
          jincluded = true
          c += 1
          if (c < children.length) {
            ci = children(c)
            assert(ci < i)
            child = regions(children(c))
          } else
            child = null
        } else {
          if (!jincluded) {
            val k = newRegionsB.length
            newRegionsB += new PSTRegion(j, j, new Array[Int](0))
            newChildren += k
          }
          j += 1
        }
      }

      val newi = newRegionsB.length
      val newr = new PSTRegion(r.start, r.end, newChildren.result())
      for (ci <- newr.children)
        newRegionsB(ci).parent = newi
      newRegionsB += newr
      regionNewRegion(i) = newi

      i += 1
    }

    val newBlocks = new Blocks(newBlocksB.result())
    val newRegions = newRegionsB.result()
    val newRoot = regionNewRegion(root)

    val newCFG = CFG(m, newBlocks)

    val pst = new PST(
      newSplitBlock.result(),
      newRegions,
      newRoot,
    )
    new PSTResult(newBlocks, newCFG, pst)
  }
}

object PST {
  def apply(m: Method, blocks: Blocks, cfg: CFG): PSTResult = {
    val pstb = new PSTBuilder(m, blocks, cfg)
    pstb.result()
  }
}

class PST(
  val splitBlock: Array[Boolean],
  val regions: Array[PSTRegion],
  val root: Int,
) {
  def nBlocks: Int = splitBlock.length

  def nRegions: Int = regions.length

  def dump(): Unit = {
    println(s"PST $nRegions:")

    def fmt(i: Int): String =
      s"${if (i > 0 && splitBlock(i - 1)) "<" else ""}$i${if (splitBlock(i)) ">" else ""}"

    println(" regions:")
    var i = 0
    while (i < nRegions) {
      val r = regions(i)
      println(s"  $i: ${fmt(r.start)} ${fmt(r.end)} ${r.parent} ${r.children.mkString(",")}")
      i += 1
    }

    println(" children:")
    def printTree(i: Int, depth: Int): Unit = {
      val r = regions(i)
      println(s"${" " * depth}$i: ${fmt(r.start)} ${fmt(r.end)}")
      for (c <- regions(i).children)
        printTree(c, depth + 2)
    }

    i = 0
    while (i < nRegions) {
      if (regions(i).parent == -1)
        printTree(i, 0)
      i += 1
    }
  }
}
