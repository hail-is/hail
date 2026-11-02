package is.hail.utils

import is.hail.utils.StackSafe._

import scala.collection.immutable.ArraySeq

import java.util.concurrent.TimeUnit

import org.openjdk.jmh.annotations.{Scope => JmhScope, _}

/** Rebuilds a tree of about 4096 leaves through `StackFrame`, adding 1 to each leaf. Each variant
  * walks children a different way, so the gap between two variants is the cost of that way.
  *
  *   - `plain`: direct recursion, no `StackFrame`. The floor.
  *   - `handLoop`: the hand-written `mapRecur` loop that #15927 deleted from `StackSafe`.
  *   - `catsTraverse`: cats `traverse` with `indexedSeqTraverse`, as in `NormalizeNames` and
  *     `Parser`.
  *   - `catsFoldM`: cats `foldM` over the child indices with an accumulator, as in
  *     `BaseIR.mapChildrenWithEnvM`.
  *
  * Run with `-prof gc` to see allocation per walk as well as time.
  */
@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(JmhScope.Thread)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
class CatsTraverseBenchmark {
  import CatsTraverseBenchmark._

  // children per node. Width 2 gives a deep, narrow tree; width 64 a shallow, wide one.
  @Param(Array("2", "8", "64"))
  var width: Int = _

  var tree: Node = _

  @Setup def setup(): Unit = {
    val depth = math.round(math.log(4096.0) / math.log(width.toDouble)).toInt
    tree = build(width, depth)
  }

  @Benchmark def plain(): Node = plainWalk(tree)

  @Benchmark def handLoop(): Node = handWalk(tree).run()

  @Benchmark def catsTraverse(): Node = traverseWalk(tree).run()

  @Benchmark def catsFoldM(): Node = foldMWalk(tree).run()
}

object CatsTraverseBenchmark {
  final class Node(val value: Int, val children: IndexedSeq[Node]) {
    def isLeaf: Boolean = children.isEmpty
  }

  def build(width: Int, depth: Int): Node =
    if (depth == 0) new Node(0, ArraySeq.empty)
    else new Node(depth, ArraySeq.fill(width)(build(width, depth - 1)))

  def plainWalk(n: Node): Node =
    if (n.isLeaf) new Node(n.value + 1, n.children)
    else new Node(n.value, n.children.map(plainWalk))

  def handWalk(n: Node): StackFrame[Node] =
    if (n.isLeaf) done(new Node(n.value + 1, n.children))
    else mapRecur(n.children)(c => call(handWalk(c))).map(new Node(n.value, _))

  def traverseWalk(n: Node): StackFrame[Node] = {
    import is.hail.collection.implicits.indexedSeqTraverse
    import cats.syntax.traverse._
    if (n.isLeaf) done(new Node(n.value + 1, n.children))
    else n.children.traverse(c => call(traverseWalk(c))).map(new Node(n.value, _))
  }

  def foldMWalk(n: Node): StackFrame[Node] = {
    import is.hail.collection.implicits.rangeTraverseOps
    if (n.isLeaf) done(new Node(n.value + 1, n.children))
    else {
      val cs = n.children.toArray
      cs.indices.foldM(0) { (acc, i) =>
        call(foldMWalk(cs(i))).map { c => cs(i) = c; acc + 1 }
      }.map(_ => new Node(n.value, ArraySeq.unsafeWrapArray(cs)))
    }
  }

  // The loop #15927 deleted from StackSafe.RichIndexedSeq, with an ArraySeq builder to match the
  // cats variants' output.
  def mapRecur[A, B](s: IndexedSeq[A])(f: A => StackFrame[B]): StackFrame[IndexedSeq[B]] = {
    val builder = ArraySeq.untagged.newBuilder[B]
    builder.sizeHint(s.size)
    var i = 0
    var cont: B => StackFrame[IndexedSeq[B]] = null
    def loop(): StackFrame[IndexedSeq[B]] =
      if (i < s.size) f(s(i)).flatMap(cont) else done(builder.result())
    cont = { b =>
      builder += b
      i += 1
      loop()
    }
    loop()
  }
}
