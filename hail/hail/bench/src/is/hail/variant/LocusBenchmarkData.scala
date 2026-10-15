package is.hail.variant

import is.hail.HailFeatureFlags
import is.hail.annotations.{Region, RegionPool}
import is.hail.asm4s.HailClassLoader
import is.hail.backend.{ExecuteContext, OwningTempFileManager}
import is.hail.expr.ir.lowering.IrMetadata
import is.hail.types.physical.{PCanonicalArray, PType}

import scala.collection.immutable.ArraySeq
import scala.collection.mutable
import scala.util.Random

/** Data and session setup shared by the per-locus benchmarks:
  *
  *   - [[ContigIndexBenchmark]]: contig index lookups, unstaged (cache helper vs reference genome)
  *     and staged (sum of packed loci over canonical loci).
  *   - [[LocusCompareBenchmark]]: comparing adjacent loci, for each pair of representations.
  *   - [[LocusCodecBenchmark]]: encoding and decoding an array of loci as structs and as varints.
  *
  * Every benchmark processes [[N]] loci per invocation and reports time and allocation per locus.
  *
  * Running, from `hail/`:
  * {{{
  *   ./mill hail.bench.runJmh 'is.hail.variant.*' -prof gc
  *   ./mill hail.bench.runJmh LocusCompareBenchmark -p switchRate=0.001 -p representation=canonical/packed
  * }}}
  * With `-prof gc`, `gc.alloc.rate.norm` is bytes allocated on the JVM heap per locus; region
  * memory, such as decoded loci, is not counted. A contig index cache hit should allocate about 0
  * bytes, so `ContigIndexBenchmark.cached` at `switchRate=0` should be about 0 B/op. To resolve
  * differences of a few percent, run more forks (`-f 3` or more) than the default one.
  *
  * Contig index cache on and off: without staged code, compare `ContigIndexBenchmark.cached` with
  * `ContigIndexBenchmark.uncached`. Staged benchmarks compile under feature flags read from the
  * environment, as a Python session would set them (see [[flagsFromEnv]]). JMH forks inherit the
  * environment, so to compare a flag's settings, such as one that turns the cache off in generated
  * code, run the same benchmark with and without its environment variable. Run mill with
  * `--no-server` when doing so: a running mill daemon keeps the environment it started with.
  * {{{
  *   SOME_FLAG_VARIABLE=1 ./mill --no-server hail.bench.runJmh ContigIndexBenchmark.stagedPacked -prof gc
  * }}}
  *
  * Comparing with another revision (for example `main`): [[LocusCompareBenchmark]] at
  * `canonical/canonical`, [[LocusCodecBenchmark]] at `encoding=struct`, and this file compile
  * against revisions without packed loci. Copy those three files into the other checkout's
  * `hail/bench/src/is/hail/variant/`, run with the same parameters on both, and compare. The
  * remaining parameter values fail in setup on such revisions; see each benchmark for its shim.
  */
object LocusBenchmarkData {

  /** Loci per invocation. A constant, so that `@OperationsPerInvocation` reports per locus. */
  final val N = 1 << 20

  /** The contigs loci are drawn from: GRCh38's chromosomes, without chrM, which is too short for
    * the positions generated here. They include contigs with a shared prefix (chr1, chr10, ...,
    * chr19) and contigs of the same length (chr2, ..., chr9, chrX, chrY), so a cache comparing
    * contig bytes cannot settle a miss by length or first byte.
    */
  val Contigs: IndexedSeq[String] =
    (1 to 22).map(i => s"chr$i") ++ ArraySeq("chrX", "chrY")

  /** [[N]] loci in runs on one contig, with positions increasing within each run. Before each locus
    * after the first, the contig switches to a different one with probability `switchRate`: 0 gives
    * one contig, 1 switches at every locus.
    */
  def loci(rg: ReferenceGenome, switchRate: Double): IndexedSeq[Locus] = {
    require(switchRate >= 0 && switchRate <= 1, switchRate)
    val rand = new Random(0)
    var contig = 0
    var position = 0
    ArraySeq.fill(N) {
      if (position > 0 && rand.nextDouble() < switchRate) {
        contig = (contig + 1 + rand.nextInt(Contigs.length - 1)) % Contigs.length
        position = 0
      }
      position += 1 + rand.nextInt(32)
      val l = Locus(Contigs(contig), position)
      require(position <= rg.contigLength(l.contig), l)
      l
    }
  }

  /** Stores `loci` in `region` as a required array of required `elementType`, returning its
    * address.
    */
  def store(ctx: ExecuteContext, elementType: PType, loci: IndexedSeq[Locus], region: Region)
    : (PCanonicalArray, Long) = {
    val pt = PCanonicalArray(elementType.setRequired(true), required = true)
    (pt, pt.unstagedStoreJavaObject(loci, region))
  }

  /** Feature flags as a Python session started in this environment would set them. Each flag is set
    * from its environment variable (as listed in [[HailFeatureFlags.defaults]]), falling back to a
    * variable named after the flag itself, which is all [[HailFeatureFlags.fromEnv]] reads.
    */
  def flagsFromEnv(env: Map[String, String] = sys.env): HailFeatureFlags =
    HailFeatureFlags.fromEnv(env ++ HailFeatureFlags.defaults.flatMap {
      case (flag, (variable, _)) =>
        Option(variable).flatMap(env.get).map(flag -> _)
    })

  /** A session with GRCh38 and flags from the environment, for compiling and running staged code on
    * one thread. Its region and region pool are its own; release them with [[closeExecuteContext]].
    *
    * It has no file system or backend: the benchmarks' staged code uses neither, and this module
    * does not compile against Hadoop or Spark.
    */
  def newExecuteContext(): ExecuteContext = {
    val references = ReferenceGenome.builtinReferences()
    new ExecuteContext(
      tmpdir = "/tmp",
      localTmpdir = "file:///tmp",
      backend = null,
      references = references,
      liftovers = Map.empty,
      sequences = Map.empty,
      fs = null,
      r = Region(pool = RegionPool()),
      tempFileManager = new OwningTempFileManager(null),
      theHailClassLoader = new HailClassLoader(getClass.getClassLoader),
      flags = flagsFromEnv(),
      irMetadata = new IrMetadata(),
      BlockMatrixCache = mutable.Map.empty,
      CompileCache = mutable.Map.empty,
      PersistedIrCache = mutable.Map.empty,
      PersistedCoercerCache = mutable.Map.empty,
    )
  }

  def closeExecuteContext(ctx: ExecuteContext): Unit = {
    ctx.close()
    ctx.r.close()
    ctx.r.pool.close()
  }

  /** Loci compare by contig index, then position. The reference ordering for checking results. */
  def lteq(rg: ReferenceGenome, l: Locus, r: Locus): Boolean = {
    val (li, ri) = (rg.getContigIndex(l.contig), rg.getContigIndex(r.contig))
    li < ri || (li == ri && l.position <= r.position)
  }

  def check[A](what: String, actual: A, expected: A): Unit =
    if (actual != expected)
      throw new AssertionError(s"$what: got $actual, expected $expected")
}
