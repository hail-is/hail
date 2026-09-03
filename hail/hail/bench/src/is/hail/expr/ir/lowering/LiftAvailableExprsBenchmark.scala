package is.hail.expr.ir.lowering

import is.hail.HailFeatureFlags
import is.hail.backend.{ExecuteContext, TempFileManager}
import is.hail.collection.{FastSeq, ImmutableMap}
import is.hail.expr.ir._
import is.hail.expr.ir.defs._
import is.hail.expr.ir.lowering.LiftAvailableExprsBenchmark._
import is.hail.types.virtual.{TBoolean, TInt32}

import scala.annotation.tailrec

import java.util.concurrent.TimeUnit

import org.openjdk.jmh.annotations.{Param, Scope => JmhScope, _}

// The value-IR shapes, one per bottleneck under study: JMH runs the cartesian
// product of shape and size, and builds each input on demand at setup.
@State(JmhScope.Thread)
class ValueShape {
  @Param(Array("bindingChain", "distinctExprs", "repeatedNests",
    "loopInvariantNest", "conditionalBusy"))
  var shape: String = _

  @Param(Array("10", "50", "100", "500"))
  var size: Int = _

  var ir: BaseIR = _

  @Setup(Level.Trial)
  def setup(): Unit =
    ir = builders(shape)(size)
}

// an order deeper than the value-IR shapes: user pipelines reach thousands of
// relational ops
@State(JmhScope.Thread)
class RelationalChain {
  @Param(Array("100", "1000", "10000", "100000"))
  var size: Int = _

  var ir: BaseIR = _

  @Setup(Level.Trial)
  def setup(): Unit =
    ir = buildRelationalChain(size)
}

@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(JmhScope.Thread)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
// the module's forkArgs don't reach JMH's forked JVM, so the stack for deep-IR
// recursion is set here
@Fork(value = 1, jvmArgsAppend = Array("-Xss32m", "-Xmx4g"))
class LiftAvailableExprsBenchmark {

  private var ctx: ExecuteContext = _

  @Setup(Level.Trial)
  def setup(): Unit =
    ctx = minimalExecuteContext

  @TearDown(Level.Trial)
  def teardown(): Unit =
    ctx.close()

  @Benchmark
  def lift(shape: ValueShape): BaseIR = LiftAvailableExprs(ctx, shape.ir)

  // a chain of n TableMapRows with a small body each: per-region setup cost
  // and the trampolined walk along the relational spine, which must hold at
  // any chain length on a production-like stack
  @Benchmark
  @Fork(value = 1, jvmArgsAppend = Array("-Xss4m", "-Xmx4g"))
  def liftRelationalChain(shape: RelationalChain): BaseIR = LiftAvailableExprs(ctx, shape.ir)
}

object LiftAvailableExprsBenchmark {

  // fresh nodes per use: the no-sharing invariant forbids reusing them
  private def a = Ref(Name("a"), TInt32)
  private def b = Ref(Name("b"), TInt32)
  private def cond = Ref(Name("cond"), TBoolean)

  private[lowering] val builders: Map[String, Int => BaseIR] =
    Map(
      "bindingChain" -> buildBindingChain,
      "distinctExprs" -> buildDistinctExprs,
      "repeatedNests" -> buildRepeatedNests,
      "loopInvariantNest" -> buildLoopInvariantNest,
      "conditionalBusy" -> buildConditionalBusy,
    )

  // n dependent bindings already in weak-ANF: per-node overhead of the two
  // phases with no motion, reuse, or hoisting
  private def buildBindingChain(n: Int): IR = {
    def go(acc: IR, k: Int): IR =
      if (k <= 0) acc
      else bindIR(acc + 1)(r => go(r, k - 1))

    go(a, n)
  }

  // n distinct compound expressions in one strict node: vnTable growth and
  // anticipated-set unions with no CSE hits
  private def buildDistinctExprs(n: Int): IR =
    maketuple((0 until n).map(i => a + I32(i)): _*)

  // n copies of one nest: every occurrence after the first collapses via
  // Vn-keyed availability (CSE)
  private def buildRepeatedNests(n: Int): IR = {
    def nest: IR = ((a + 1) * (b + 2)) + ((a + 1) * (b + 2))

    maketuple(Seq.fill(n)(nest): _*)
  }

  // a depth-n invariant nest inside a stream lambda: every level hoists
  // across the loop frame (LICM)
  private def buildLoopInvariantNest(n: Int): IR = {
    def deep(k: Int): IR =
      if (k <= 0) a
      else deep(k - 1) + k

    ToArray(mapIR(rangeIR(10))(elt => elt + deep(n)))
  }

  // n Ifs each with a total value busy in both branches: the busy arm of the
  // anticipated sets licenses a hoist above each conditional (PRE)
  private def buildConditionalBusy(n: Int): IR = {
    def go(k: Int): IR =
      if (k <= 0) a
      else bindIR(If(cond, a + k, (a + k) * (b + 1)))(x => go(k - 1) + x)

    go(n)
  }

  private[lowering] def buildRelationalChain(n: Int): BaseIR = {
    @tailrec def go(t: TableIR, k: Int): TableIR =
      if (k <= 0) t
      else go(
        t.mapRows((_, row) => InsertFields(row, FastSeq("x" -> (GetField(row, "idx") + 1)))),
        k - 1,
      )

    go(TableRange(10, 1), n)
  }

  // the pass reads only irMetadata and (gated-off) invariant flags, so the
  // backend, filesystem, and region slots can stay empty
  private[lowering] def minimalExecuteContext: ExecuteContext =
    new ExecuteContext(
      tmpdir = "/tmp",
      localTmpdir = "file:///tmp",
      backend = null,
      references = Map.empty,
      fs = null,
      r = null,
      tempFileManager = NoTempFiles,
      theHailClassLoader = null,
      flags = HailFeatureFlags.fromEnv(Map.empty),
      irMetadata = new IrMetadata(),
      BlockMatrixCache = ImmutableMap.empty,
      CompileCache = ImmutableMap.empty,
      PersistedIrCache = ImmutableMap.empty,
      PersistedCoercerCache = ImmutableMap.empty,
    )

  private object NoTempFiles extends TempFileManager {
    override def newTmpPath(tmpdir: String, prefix: String, extension: String): String =
      throw new UnsupportedOperationException("benchmarks may not create temp files")

    override def close(): Unit = ()
  }
}
