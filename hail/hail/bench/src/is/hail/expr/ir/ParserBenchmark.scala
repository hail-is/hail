package is.hail.expr.ir

import is.hail.{HailFeatureFlags, SharedResources}
import is.hail.annotations.Region
import is.hail.backend.{ExecuteContext, OwningTempFileManager}
import is.hail.backend.local.LocalTaskContext
import is.hail.collection.ImmutableMap
import is.hail.expr.ir.lowering.IrMetadata
import is.hail.io.reference.RgAttachments
import is.hail.scalacheck.ArbitraryIR
import is.hail.utils.lazily

import scala.collection.mutable

import java.util.concurrent.TimeUnit

import org.openjdk.jmh.annotations.{Scope => JmhScope, _}
import org.scalacheck.Gen
import org.scalacheck.rng.Seed

@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(JmhScope.Thread)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
class ParserBenchmark {
  @Param(Array("1000", "10000", "100000"))
  var size: Int = _

  var shared: SharedResources = _
  var taskContext: LocalTaskContext = _
  var ctx: ExecuteContext = _
  var text: String = _

  @Setup def setup(): Unit = {
    shared = new SharedResources()
    taskContext = new LocalTaskContext(0, 0)
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
      flags = HailFeatureFlags.fromEnv(Map.empty),
      irMetadata = new IrMetadata(),
      BlockMatrixCache = ImmutableMap.empty,
      CompileCache = ImmutableMap.empty,
      PersistedIrCache = ImmutableMap.empty,
      PersistedCoercerCache = ImmutableMap.empty,
      taskContext = taskContext,
      memo = mutable.HashMap.empty,
      fsBc = lazily(shared.fsBc),
    )
    val gen =
      for {
        t <- ArbitraryIR.genType()
        ir <- ArbitraryIR.genIR(t, BindingEnv.empty, size)
      } yield ir
    val ir = gen.pureApply(Gen.Parameters.default, Seed(size.toLong))
    TypeCheck(ctx, ir)
    text = Pretty.sexprStyle(ir, elideLiterals = false)
    assert(IRParser.parse_value_ir(ctx, text) == ir, "parsing the printed IR gave a different IR")
  }

  @TearDown def tearDown(): Unit = {
    ctx.close()
    taskContext.close()
    shared.close()
  }

  @Benchmark def parse(): IR =
    IRParser.parse(text, IRParser.ir_value_expr(ctx, BindingEnv.empty)(_).run())
}
