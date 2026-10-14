package is.hail.variant

import is.hail.asm4s._
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.types.physical.{PCanonicalArray, PCanonicalLocus}
import is.hail.variant.LocusBenchmarkData._

import java.util.concurrent.TimeUnit

import org.openjdk.jmh.annotations.{Scope => JmhScope, _}

/** Converting canonical loci, whose contig is a string, to contig indices. See
  * [[LocusBenchmarkData]] for how to run these and what they share.
  *
  *   - `cached` and `uncached` look up the contig of each locus without staged code: `cached` hands
  *     a [[ContigIndexCache]] the contig's bytes in the region, `uncached` loads the contig as a
  *     String and asks the reference genome. Their difference is the cache's effect.
  *   - `stagedPacked` sums the packed integers of an array of canonical loci in generated code,
  *     which looks up each contig index the way generated code does in this session. To measure
  *     generated code without the cache, run it with the cache turned off through a feature flag's
  *     environment variable, where the session has such a flag.
  */
@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(JmhScope.Thread)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
@OperationsPerInvocation(LocusBenchmarkData.N)
class ContigIndexBenchmark {

  @Param(Array("0", "0.001", "0.5", "1.0"))
  var switchRate: Double = _

  private var ctx: ExecuteContext = _
  private var rg: ReferenceGenome = _
  private val locusType = PCanonicalLocus(ReferenceGenome.GRCh38)
  private var arrayType: PCanonicalArray = _
  private var array: Long = _
  private var contigAddresses: Array[Long] = _
  private var cache: ContigIndexCache = _
  private var sumPacked: AsmFunction1[Long, Long] = _

  @Setup(Level.Trial)
  def setup(): Unit = {
    ctx = newExecuteContext()
    rg = ctx.references(ReferenceGenome.GRCh38)
    val data = loci(rg, switchRate)
    val (pt, addr) = store(ctx, locusType, data, ctx.r)
    arrayType = pt
    array = addr
    contigAddresses = Array.tabulate(N) { i =>
      locusType.representation.loadField(arrayType.loadElement(array, N, i), 0)
    }
    cache = new ContigIndexCache(rg)
    sumPacked = compileSumPacked()

    val indices = data.map(l => rg.getContigIndex(l.contig))
    check("cached", cached(), indices.map(_.toLong).sum)
    check("uncached", uncached(), indices.map(_.toLong).sum)
    check(
      "stagedPacked",
      stagedPacked(),
      data.lazyZip(indices).map((l, i) => (i.toLong << 32) | l.position.toLong).sum,
    )
  }

  @TearDown(Level.Trial)
  def tearDown(): Unit =
    closeExecuteContext(ctx)

  private def compileSumPacked(): AsmFunction1[Long, Long] = {
    val fb = EmitFunctionBuilder[Long, Long](ctx, "sum_packed")
    fb.emitWithBuilder { cb =>
      val a = arrayType.loadCheapSCode(cb, fb.getCodeParam[Long](1)).asIndexable
      val sum = cb.newLocal[Long]("sum", 0L)
      val i = cb.newLocal[Int]("i")
      cb.for_(
        cb.assign(i, 0),
        i < a.loadLength,
        cb.assign(i, i + 1),
        cb.assign(sum, sum + a.loadElement(cb, i).getOrAssert(cb).asLocus.packed(cb)),
      )
      sum
    }
    fb.resultWithIndex()(ctx.theHailClassLoader, ctx.fs, ctx.taskContext, ctx.r)
  }

  @Benchmark
  def cached(): Long = {
    val binary = locusType.contigType.binaryRepresentation
    var sum = 0L
    var i = 0
    while (i < contigAddresses.length) {
      val c = contigAddresses(i)
      sum += cache.contigIndex(binary.bytesAddress(c), binary.loadLength(c))
      i += 1
    }
    sum
  }

  @Benchmark
  def uncached(): Long = {
    val contigType = locusType.contigType
    var sum = 0L
    var i = 0
    while (i < contigAddresses.length) {
      sum += rg.getContigIndex(contigType.loadString(contigAddresses(i)))
      i += 1
    }
    sum
  }

  @Benchmark
  def stagedPacked(): Long =
    sumPacked(array)
}
