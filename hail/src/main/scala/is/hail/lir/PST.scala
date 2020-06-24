package is.hail.lir

import is.hail.utils.ArrayBuilder

import scala.collection.mutable

object Region {
  implicit val ord: Ordering[Region] = new Ordering[Region] {
    def compare(r1: Region, r2: Region): Int = {
      val c = Integer.compare(r1.start, r2.start)
      if (c != 0)
        return c

      -Integer.compare(r1.end, r2.end)
    }
  }
}

class Region(
  val start: Int,
  val end: Int) {
  // computed
  var parent: Int = -1
  var children: Array[Int] = null
}

object PST {
  def apply(cfg: CFG): PST = {
    val nBlocks = cfg.nBlocks

    // identify back edges
    val backEdges = mutable.Set[(Int, Int)]()

    {
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

    // linear blocks
    val linearization = new Array[Int](nBlocks)

    // successors
    // in linear index
    val forwardSucc = Array.fill(nBlocks)(mutable.Set[Int]())
    val backwardSucc = Array.fill(nBlocks)(mutable.Set[Int]())
    // predecessors
    val backwardPred = Array.fill(nBlocks)(mutable.Set[Int]())

    val blockLinearIdx = new Array[Int](nBlocks)

    {
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

    // for debugging
    def checkRegion(start: Int, end: Int): Unit = {
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

    // maxTargetLE(i) is the largest back edge target <= i
    // from a back edge with source greater than i
    val maxTargetLE = new Array[Int](nBlocks)

    {
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
    val minSourceGE = new Array[Int](nBlocks)

    {
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

    def backEdgesOK(start: Int, end: Int): Boolean = {
      val maxTarget = maxTargetLE(end)
      val minSource = minSourceGE(start)
      (maxTarget == -1 || maxTarget <= start) &&
        (minSource == -1 || minSource >= end)
    }

    val regionsb = new ArrayBuilder[Region]()

    // find regions in [start, max]
    // [start, end] is a region
    def findRegions(start: Int, end: Int): Unit = {
      var regionStarts = new ArrayBuilder[Int]()
      regionStarts += start

      // find subregions of [start, end]

      // forward edge targets from [start, newStart) into [newStart, end]
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
                regionsb += new Region(rStart, newStart)
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

    findRegions(0, nBlocks - 1)

    // compute tree
    val regions = regionsb.result().sorted
    val nRegions = regions.length

    {
      var n = 0
      def findChildren(): Unit = {
        val i = n
        val r = regions(i)
        n += 1

        val childrenb = new ArrayBuilder[Int]()

        val prev: Region = null

        @scala.annotation.tailrec
        def f(): Unit = {
          if (n < nRegions) {
            val next = regions(n)
            assert(r.start <= next.start)
            if (next.end <= r.end) {
              if (prev != null) {
                assert(prev.end <= next.start)
              }
              childrenb += n
              assert(next.parent == -1)
              next.parent = i
              findChildren()
              f()
            } else
              assert(next.start >= r.end)
          }
        }
        f()

        r.children = childrenb.result()
      }
      while (n < nRegions) {
        findChildren()
      }
      assert(n == nRegions)
    }

    new PST(linearization, blockLinearIdx, regions)
  }
}

class  PST(
  val linearization: Array[Int],
  val blockLinearIdx: Array[Int],
  val regions: Array[Region]
) {
  def nBlocks: Int = linearization.length

  def nRegions: Int = regions.length

  def dump(): Unit = {
    println(s"PST $nBlocks $nRegions:")

    println(" linearization:")
    var i = 0
    while (i < nBlocks) {
      println(s"  $i ${ linearization(i) }")
      i += 1
    }

    println(" regions:")
    i = 0
    while (i < nRegions) {
      val r = regions(i)
      println(s"  $i: ${ r.start } ${ r.end } [${ linearization(r.start) } ${ linearization(r.end) }] ${ r.parent } ${ r.children.mkString(",") }")
      i += 1
    }

    println(" children:")
    def printTree(i: Int, depth: Int): Unit = {
      val r = regions(i)
      println(s"${ " " * depth }$i: ${ r.start } ${ r.end } [${ linearization(r.start)} ${ linearization(r.end) }]")
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
