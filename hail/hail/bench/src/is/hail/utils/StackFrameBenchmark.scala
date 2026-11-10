package is.hail.utils

import is.hail.utils.StackSafe._

import scala.collection.immutable.ArraySeq

import java.util.concurrent.TimeUnit

import org.openjdk.jmh.annotations.{Scope => JmhScope, _}

/** Rebuilds a tree of about 4096 leaves, adding 1 to each leaf.
  *
  *   - `plain`: direct recursion, no `StackFrame`. The floor.
  *   - `mapRecur`: recursion through `StackFrame` with `mapRecur`, as in the IR walkers.
  *
  * Run with `-prof gc` to see allocation per walk as well as time.
  */
@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(JmhScope.Thread)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
class StackFrameBenchmark {
  import StackFrameBenchmark._

  // children per node. Width 2 gives a deep, narrow tree; width 64 a shallow, wide one.
  @Param(Array("2", "8", "64"))
  var width: Int = _

  var tree: Node = _

  @Setup def setup(): Unit = {
    val depth = math.round(math.log(4096.0) / math.log(width.toDouble)).toInt
    tree = build(width, depth)
  }

  @Benchmark def plain(): Node = plainWalk(tree)

  @Benchmark def mapRecur(): Node = mapRecurWalk(tree).run()
}

object StackFrameBenchmark {
  final class Node(val value: Int, val children: IndexedSeq[Node]) {
    def isLeaf: Boolean = children.isEmpty
  }

  def build(width: Int, depth: Int): Node =
    if (depth == 0) new Node(0, ArraySeq.empty)
    else new Node(depth, ArraySeq.fill(width)(build(width, depth - 1)))

  def plainWalk(n: Node): Node =
    if (n.isLeaf) new Node(n.value + 1, n.children)
    else new Node(n.value, n.children.map(plainWalk))

  def mapRecurWalk(n: Node): StackFrame[Node] =
    if (n.isLeaf) done(new Node(n.value + 1, n.children))
    else n.children.mapRecur(mapRecurWalk).map(cs => new Node(n.value, cs))
}
