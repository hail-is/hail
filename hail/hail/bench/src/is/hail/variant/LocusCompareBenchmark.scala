package is.hail.variant

import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.types.physical.{PCanonicalArray, PCanonicalLocus, PPackedLocus, PType}
import is.hail.variant.LocusBenchmarkData._

import java.util.concurrent.TimeUnit

import org.openjdk.jmh.annotations.{Scope => JmhScope, _}

/** Comparing loci in generated code: counts the adjacent pairs `(left(i), right(i + 1))` that are
  * in order, where `left` and `right` hold the same loci in the representations named by
  * `representation`. See [[LocusBenchmarkData]] for how to run this and what it shares.
  *
  * Compatibility shim: the packed locus type is parsed from its name instead of referenced, so that
  * this file compiles on revisions without it, where `canonical/canonical` is the baseline. On
  * those revisions, the other representations fail in setup.
  */
@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(JmhScope.Thread)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
@OperationsPerInvocation(LocusBenchmarkData.N)
class LocusCompareBenchmark {

  @Param(Array("0", "0.001", "0.5", "1.0"))
  var switchRate: Double = _

  @Param(Array("canonical/canonical", "packed/packed", "canonical/packed"))
  var representation: String = _

  private var ctx: ExecuteContext = _
  private var left: Long = _
  private var right: Long = _
  private var countInOrder: AsmFunction2[Long, Long, Long] = _

  private def locusType(name: String, rg: ReferenceGenome): PType = name match {
    case "canonical" => PCanonicalLocus(rg)
    case "packed" => PPackedLocus(rg)
  }

  @Setup(Level.Trial)
  def setup(): Unit = {
    ctx = newExecuteContext()
    val rg = ctx.references(ReferenceGenome.GRCh38)
    val data = loci(rg, switchRate)
    val Array(leftName, rightName) = representation.split('/')
    val (leftType, leftAddr) = store(ctx, locusType(leftName, rg), data, ctx.r)
    val (rightType, rightAddr) = store(ctx, locusType(rightName, rg), data, ctx.r)
    left = leftAddr
    right = rightAddr
    countInOrder = compile(leftType, rightType)

    check(
      "countInOrder",
      countInOrderPairs(),
      (0 until N - 1).count(i => lteq(rg, data(i), data(i + 1))).toLong,
    )
  }

  @TearDown(Level.Trial)
  def tearDown(): Unit =
    closeExecuteContext(ctx)

  private def compile(leftType: PCanonicalArray, rightType: PCanonicalArray)
    : AsmFunction2[Long, Long, Long] = {
    val fb = EmitFunctionBuilder[Long, Long, Long](ctx, "count_in_order")
    fb.emitWithBuilder { cb =>
      val l = leftType.loadCheapSCode(cb, fb.getCodeParam[Long](1)).asIndexable
      val r = rightType.loadCheapSCode(cb, fb.getCodeParam[Long](2)).asIndexable
      val ord = fb.ecb.getOrdering(leftType.elementType.sType, rightType.elementType.sType)
      val count = cb.newLocal[Long]("count", 0L)
      val i = cb.newLocal[Int]("i")
      cb.for_(
        cb.assign(i, 0),
        i < l.loadLength - 1,
        cb.assign(i, i + 1), {
          val x = l.loadElement(cb, i).getOrAssert(cb)
          val y = r.loadElement(cb, i + 1).getOrAssert(cb)
          cb.if_(ord.lteqNonnull(cb, x, y), cb.assign(count, count + 1L))
        },
      )
      count
    }
    fb.resultWithIndex()(ctx.theHailClassLoader, ctx.fs, ctx.taskContext, ctx.r)
  }

  @Benchmark
  def countInOrderPairs(): Long =
    countInOrder(left, right)
}
