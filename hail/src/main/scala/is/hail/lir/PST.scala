package is.hail.lir

import is.hail.utils.ArrayBuilder

import scala.collection.mutable

class Region(
  var start: Int,
  var end: Int,
  var children: Array[Int],
  // -1 means root, or parent not yet known
  var parent: Int = -1)

class PSTBuilder(
  m: Method,
  blocks: Blocks,
  cfg: CFG
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
    val stack = mutable.Stack[(Int, Iterator[Int])]()
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
      for (p <- cfg.pred(i)) {
        if (!backEdges(p -> i))
          n += 1
      }
      n
    }
    var k = 0

    // recursion will blow out the stack
    val stack = mutable.Stack[(Int, Iterator[Int])]()

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

  private val regions: mutable.ArrayBuffer[Region] = mutable.ArrayBuffer[Region]()

  // regions with no parents
  private val frontier = new ArrayBuilder[Int]()

  private def addRegion(start: Int, end: Int): Int = {
    var firstc = frontier.length
    while ((firstc - 1) >= 0 && regions(frontier(firstc - 1)).start >= start)
      firstc -= 1
    assert(firstc == 0 || regions(frontier(firstc - 1)).end <= start)

    val ri = regions.length
    val n = frontier.length - firstc
    val children = new Array[Int](n)
    var i = 0
    while (i < n) {
      val c = frontier(firstc + i)
      assert(regions(c).parent == -1)
      regions(c).parent = ri
      children(i) = c
      i += 1
    }
    frontier.setSizeUninitialized(frontier.length - n)
    if (frontier.nonEmpty && regions(frontier.last).end == start)
      splitBlock.set(start)
    frontier += ri
    regions += new Region(start, end, children)
    ri
  }

  private def addRoot(): Int = {
    if (frontier.length == 1 &&
      regions(frontier(0)).start == 0 &&
      regions(frontier(0)).end == nBlocks - 1) {
      frontier(0)
    } else {
      val c = regions.length

      val ri = regions.length
      val n = frontier.length
      val children = new Array[Int](n)
      var i = 0
      while (i < n) {
        val c = frontier(i)
        assert(regions(c).parent == -1)
        regions(c).parent = ri
        children(i) = c
        i += 1
      }
      regions += new Region(0, nBlocks - 1, children)
      frontier.clear()
      frontier += c
      c
    }
  }

  // find regions in [start, end]
  // no edges from [0, start) target (start, end]
  private def findRegions(start: Int, end: Int): Unit = {
    var regionStarts = new ArrayBuilder[Int]()
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

        f(regionStarts.length - 1)

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

  def result(): PST = {
    computeBackEdges()
    linearize()
    computeMaxTargetLE()
    computeMinSourceGE()
    findRegions(0, nBlocks - 1)
    val root = addRoot()

    val newBlocks = new ArrayBuilder[Block]()
    val newSplitBlock = new ArrayBuilder[Boolean]()

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
        val newi = newBlocks.length
        newBlocks += b
        newSplitBlock += true
        newBlocks += splitb
        newSplitBlock += false
        blockNewEndBlockIdx(i) = newi
        blockNewStartBlockIdx(i) = newi + 1
      } else {
        val newi = newBlocks.length
        newBlocks += b
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
    val newRegions = new ArrayBuilder[Region]()
    val regionNewRegion = new Array[Int](regions.length)
    i = 0
    while (i < regions.length) {
      val r = regions(i)
      val children = r.children

      var c = 0
      var ci = 0
      var child: Region = null
      if (c < children.length) {
        ci = children(c)
        assert(ci < i)
        child = regions(children(c))
      }

      val newChildren = new ArrayBuilder[Int]()

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
            val k = newRegions.length
            newRegions += new Region(j, j, new Array[Int](0))
            newChildren += k
          }
          j += 1
        }
      }

      val newi = newRegions.length
      val newr = new Region(r.start, r.end, newChildren.result())
      for (ci <- newr.children)
        newRegions(ci).parent = newi
      newRegions += newr
      regionNewRegion(i) = newi

      i += 1
    }

    new PST(
      new Blocks(newBlocks.result()),
      newSplitBlock.result(),
      newRegions.result(),
      regionNewRegion(root))
  }
}

object PST {
  def apply(m: Method, blocks: Blocks, cfg: CFG): PST = {
    val pstb = new PSTBuilder(m, blocks, cfg)
    pstb.result()
  }
}

class  PST(
  val blocks: Blocks,
  val splitBlock: Array[Boolean],
  val regions: Array[Region],
  val root: Int
) {
  assert(blocks.length == splitBlock.length)

  def nBlocks: Int = blocks.length

  def nRegions: Int = regions.length

  def dump(): Unit = {
    println(s"PST $nBlocks $nRegions:")

    def fmt(i: Int): String =
      s"${ if (i > 0 && splitBlock(i - 1)) "<" else "" }$i${ if (splitBlock(i)) ">" else "" }"

    println(" regions:")
    var i = 0
    while (i < nRegions) {
      val r = regions(i)
      println(s"  $i: ${ fmt(r.start) } ${ fmt(r.end) } ${ r.parent } ${ r.children.mkString(",") }")
      i += 1
    }

    println(" children:")
    def printTree(i: Int, depth: Int): Unit = {
      val r = regions(i)
      println(s"${ " " * depth }$i: ${ fmt(r.start) } ${ fmt(r.end) }")
      for (c <- regions(i).children) {
        printTree(c, depth + 2)
      }
    }

    i = 0
    while (i < nRegions) {
      if (regions(i).parent == -1)
        printTree(i, 0)
      i += 1
    }
  }
}
