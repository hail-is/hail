package is.hail.expr.ir

import is.hail.collection.FastSeq
import is.hail.expr.ir.analyses.{ControlFlowPreventsSplit => Trace}
import is.hail.expr.ir.defs.{I32, Recur, Ref, TailLoop}
import is.hail.types.virtual.{TInt32, TStream}

import java.util.concurrent.TimeUnit

import org.openjdk.jmh.annotations.{Scope => JmhScope, _}

@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(JmhScope.Thread)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
class ControlFlowPreventsSplitsAlternatives {
  import ControlFlowPreventsSplitsAlternatives._

  @Param(Array("10", "50", "100", "500"))
  var depth: Int = _

  private var tree: IR = _
  private var usesAndDefs: UsesAndDefs = _

  @Setup(Level.Trial)
  def setup(): Unit = {
    tree = buildTailLoopTree(depth)
    usesAndDefs = ComputeUsesAndDefs(tree, errorIfFreeVariables = false)
  }

  @Benchmark
  def parentPointers(): Memo[Unit] = {
    val pp = ParentPointers(tree)
    ControlFlowPreventsSplit(tree, pp, usesAndDefs)
  }

  @Benchmark
  def trace(): Memo[Unit] =
    Trace(tree, usesAndDefs)
}

object ControlFlowPreventsSplitsAlternatives {

  private def buildTailLoopTree(depth: Int): IR = {
    tailLoop(TInt32, I32(0)) { case (recur, Seq(k)) =>
      def chain(acc: IR, n: Int): IR =
        if (n <= 0) recur(FastSeq(acc))
        else bindIR(acc + 1)(acc => chain(acc, n - 1))

      chain(k, depth)
    }
  }

  def ParentPointers(x: BaseIR): Memo[BaseIR] = {
    val m = Memo.empty[BaseIR]

    def recur(ir: BaseIR, parent: BaseIR): Unit = {
      m.bind(ir, parent)
      ir.children.foreach(recur(_, ir))
    }

    recur(x, null)
    m
  }

  def ControlFlowPreventsSplit(x: BaseIR, parentPointers: Memo[BaseIR], usesAndDefs: UsesAndDefs)
    : Memo[Unit] = {
    val m = Memo.empty[Unit]
    VisitIR(x) {
      case r @ Recur(name, _, _) =>
        var parent: BaseIR = r
        while (
          parent match {
            case TailLoop(`name`, _, _, _) => false
            case _ => true
          }
        ) {
          if (!m.contains(parent))
            m.bind(parent, ())
          parent = parentPointers.lookup(parent)
        }
      case r @ Ref(_, t) if t.isInstanceOf[TStream] =>
        val declaration = usesAndDefs.defs.lookup(r)
        var parent: BaseIR = r
        while (!parent.eq(declaration)) {
          if (!m.contains(parent))
            m.bind(parent, ())
          parent = parentPointers.lookup(parent)
        }
      case _ =>
    }
    m
  }
}
