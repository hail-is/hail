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
  * end. It uses only long-standing entry points, so it can be copied into an older checkout to
  * compare.
  *
  *   - `lets`: one `Block` of 3000 bindings, each using the one before.
  *   - `deep`: 2000 nested `StreamMap`s, a long first-child chain.
  *   - `tree`: a balanced tree of 16383 `Add` nodes.
  *   - `wide`: one `MakeStruct` of 1000 fields, each reading a `Block` binding, so renaming changes
  *     every child.
  */
@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(JmhScope.Thread)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
class IRPassBenchmark {
  @Param(Array("lets", "deep", "tree", "wide"))
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

  @Benchmark def normalizeNames(): BaseIR =
    NormalizeNames()(ctx, ir)

  @Benchmark def typeCheck(): Unit =
    TypeCheck(ctx, ir)

  @Benchmark def forwardLets(): BaseIR =
    ForwardLets(ctx, ir)

  @Benchmark def rewriteBottomUp(): BaseIR =
    RewriteBottomUp(ir, _ => None)
}

object IRPassBenchmark {
  def build(shape: String): IR =
    shape match {
      case "lets" => lets(3000)
      case "deep" => ToArray(deep(2000))
      case "tree" => tree(13)
      case "wide" => wide(1000)
    }

  private def lets(n: Int): IR = {
    val bindings =
      (1 until n)
        .scanLeft(Binding(Name("x0"), I32(0))) { (prev, i) =>
          Binding(Name(s"x$i"), Ref(prev.name, TInt32) + i)
        }

    Block(bindings, Ref(bindings.last.name, TInt32))
  }

  private def deep(d: Int): IR =
    if (d == 0) StreamRange(I32(0), I32(10), I32(1))
    else deep(d - 1).streamMap(_ + I32(d))

  private def tree(depth: Int): IR =
    if (depth == 0) I32(1)
    else tree(depth - 1) + tree(depth - 1)

  private def wide(n: Int): IR =
    bindIR(I32(0))(x => MakeStruct((0 until n).map(i => s"f$i" -> (x + i))))
}
