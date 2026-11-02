package is.hail.expr.ir

import is.hail.{HailFeatureFlags, SharedResources}
import is.hail.annotations.Region
import is.hail.backend.{ExecuteContext, OwningTempFileManager}
import is.hail.collection.ImmutableMap
import is.hail.expr.ir.defs._
import is.hail.expr.ir.lowering.{ForwardLets, IrMetadata, RewriteBottomUp}
import is.hail.io.reference.RgAttachments
import is.hail.types.virtual.TInt32

import java.util.concurrent.TimeUnit

import org.openjdk.jmh.annotations.{Scope => JmhScope, _}

/** Times whole compiler passes on large synthetic IR, to see whether walker changes show up end to
  * end. Copy this file into each checkout you want to compare; it uses only entry points that are
  * the same before and after #15927.
  *
  *   - `lets`: one `Block` of 3000 bindings, each using the one before.
  *   - `deep`: 2000 nested `StreamMap`s, a long first-child chain.
  *   - `tree`: a balanced tree of 16383 `Add` nodes.
  */
@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(JmhScope.Thread)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
class IRPassBenchmark {
  @Param(Array("lets", "deep", "tree"))
  var shape: String = _

  var shared: SharedResources = _
  var ctx: ExecuteContext = _
  var ir: IR = _

  @Setup def setup(): Unit = {
    shared = new SharedResources()
    ctx = new ExecuteContext(
      tmpdir = "/tmp",
      localTmpdir = "file:///tmp",
      backend = shared.backend,
      references = shared.references,
      attachments = RgAttachments.empty,
      fs = shared.fs,
      r = Region(pool = shared.pool),
      tempFileManager = new OwningTempFileManager(shared.fs),
      theHailClassLoader = shared.hcl,
      // defaults, so no strict invariants: production doesn't run the extra verifying walks
      flags = HailFeatureFlags.fromEnv(Map.empty),
      irMetadata = new IrMetadata(),
      BlockMatrixCache = ImmutableMap.empty,
      CompileCache = ImmutableMap.empty,
      PersistedIrCache = ImmutableMap.empty,
      PersistedCoercerCache = ImmutableMap.empty,
    )
    ir = IRPassBenchmark.build(shape)
  }

  @TearDown def tearDown(): Unit = {
    ctx.close()
    shared.close()
  }

  @Benchmark def normalizeNames(): BaseIR = NormalizeNames()(ctx, ir)

  @Benchmark def typeCheck(): Unit = TypeCheck(ctx, ir)

  @Benchmark def forwardLets(): BaseIR = ForwardLets(ctx, ir)

  @Benchmark def rewriteBottomUp(): BaseIR = RewriteBottomUp(ir, _ => None)
}

object IRPassBenchmark {
  def build(shape: String): IR = shape match {
    case "lets" => lets(3000)
    case "deep" => ToArray(deep(2000))
    case "tree" => tree(13)
  }

  private def add(l: IR, r: IR): IR = ApplyBinaryPrimOp(Add(), l, r)

  private def lets(n: Int): IR = {
    val names = (0 until n).map(i => Name(s"x$i"))
    val bindings = (0 until n).map { i =>
      Binding(names(i), if (i == 0) I32(0) else add(Ref(names(i - 1), TInt32), I32(i)))
    }
    Block(bindings, Ref(names(n - 1), TInt32))
  }

  private def deep(d: Int): IR =
    if (d == 0) StreamRange(I32(0), I32(10), I32(1))
    else {
      val n = Name(s"s$d")
      StreamMap(deep(d - 1), n, add(Ref(n, TInt32), I32(d)))
    }

  private def tree(depth: Int): IR =
    if (depth == 0) I32(1) else add(tree(depth - 1), tree(depth - 1))
}
