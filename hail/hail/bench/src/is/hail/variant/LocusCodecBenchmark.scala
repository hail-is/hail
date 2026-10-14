package is.hail.variant

import is.hail.annotations.Region
import is.hail.backend.ExecuteContext
import is.hail.io.BufferSpec
import is.hail.types.encoded._
import is.hail.types.physical.{PCanonicalArray, PCanonicalLocus}
import is.hail.types.virtual.{TArray, TLocus}
import is.hail.variant.LocusBenchmarkData._

import scala.collection.immutable.ArraySeq

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}
import java.util.concurrent.TimeUnit

import org.openjdk.jmh.annotations.{Scope => JmhScope, _}

/** Encoding an array of canonical loci, and decoding it, in memory and without compression.
  * `encoding=struct` encodes each locus as a struct of contig and position, and decodes it to a
  * canonical locus. `encoding=varint` encodes each locus as its packed integer, and decodes it to a
  * packed locus. Encoding canonical loci as varints computes each locus's contig index, so `encode`
  * at `encoding=varint` also depends on how contig indices are looked up. See
  * [[LocusBenchmarkData]] for how to run this and what it shares.
  *
  * Compatibility shim: decoded loci are read without naming the packed locus type, whose element is
  * read as its 8-byte integer, so that this file compiles on revisions without packed loci, where
  * `encoding=struct` is the baseline. On those revisions, `encoding=varint` fails in setup.
  */
@BenchmarkMode(Array(Mode.AverageTime))
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(JmhScope.Thread)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
@OperationsPerInvocation(LocusBenchmarkData.N)
class LocusCodecBenchmark {

  @Param(Array("0", "0.001", "0.5", "1.0"))
  var switchRate: Double = _

  @Param(Array("struct", "varint"))
  var encoding: String = _

  private val spec = BufferSpec.blockedUncompressed

  private var ctx: ExecuteContext = _
  private var array: Long = _
  private var encoder: EncoderAsmFunction = _
  private var decoder: DecoderAsmFunction = _
  private var encodedOut: ByteArrayOutputStream = _
  private var encoded: Array[Byte] = _
  private var decodeRegion: Region = _

  private def locusEType: EType = encoding match {
    case "struct" =>
      EBaseStruct(
        ArraySeq(EField("contig", EBinary(true), 0), EField("position", EInt32(true), 1)),
        required = true,
      )
    case "varint" => EVarint(true)
  }

  @Setup(Level.Trial)
  def setup(): Unit = {
    ctx = newExecuteContext()
    val rg = ctx.references(ReferenceGenome.GRCh38)
    val data = loci(rg, switchRate)
    val (arrayType, addr) = store(ctx, PCanonicalLocus(rg.name), data, ctx.r)
    array = addr
    val et = EArray(locusEType, required = true)
    encoder = EType.buildEncoder(ctx, et, arrayType)(ctx.theHailClassLoader)
    val (decodedArrayType, buildDecoder) =
      EType.buildDecoderToRegionValue(ctx, et, TArray(TLocus(rg.name)))
    decoder = buildDecoder(ctx.theHailClassLoader)
    encodedOut = new ByteArrayOutputStream()
    decodeRegion = Region(pool = ctx.r.pool)

    encode(): Unit
    encoded = encodedOut.toByteArray
    val decoded = decode()
    val decodedType = decodedArrayType.asInstanceOf[PCanonicalArray]
    check("decoded length", decodedType.loadLength(decoded), N)
    val readLocus: Long => Locus = (decodedType.elementType, encoding) match {
      case (t: PCanonicalLocus, "struct") =>
        a => Locus(t.contigType.loadString(t.representation.loadField(a, 0)), t.position(a))
      case (t, "varint") if !t.isInstanceOf[PCanonicalLocus] && t.byteSize == 8 =>
        a => {
          val packed = Region.loadLong(a)
          Locus(rg.contigs((packed >>> 32).toInt), packed.toInt)
        }
      case (t, _) =>
        throw new AssertionError(s"decoded $encoding to unexpected locus type $t")
    }
    (0 until N).foreach { i =>
      check(s"decoded locus $i", readLocus(decodedType.loadElement(decoded, N, i)), data(i))
    }
  }

  @TearDown(Level.Trial)
  def tearDown(): Unit = {
    decodeRegion.close()
    closeExecuteContext(ctx)
  }

  @Benchmark
  def encode(): Int = {
    encodedOut.reset()
    val out = spec.buildOutputBuffer(encodedOut)
    encoder(array, out)
    out.flush()
    encodedOut.size()
  }

  @Benchmark
  def decode(): Long = {
    decodeRegion.clear()
    decoder(decodeRegion, spec.buildInputBuffer(new ByteArrayInputStream(encoded)))
  }
}
