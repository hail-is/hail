package is.hail.io.bgen

import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir.{invoke, Env, MatrixRead, MatrixRowsTable, TableKeyBy}
import is.hail.expr.ir.defs.{False, I32, Literal, Str, TableCollect}
import is.hail.types.physical.LocusRepresentations.forEachCanonicalLocus
import is.hail.types.virtual._
import is.hail.utils.using
import is.hail.variant.BuiltinLoci.tlocus

import scala.collection.immutable.ArraySeq

import org.junit.jupiter.api.Test

object BgenIndexSuite {

  /** Indexes `bgen` on GRCh37, writing the index to `idx`. */
  def indexBgen(
    bgen: String,
    idx: String,
    contigRecoding: Map[String, String],
  )(implicit
    ctx: ExecuteContext
  ): Unit =
    eval(
      invoke(
        "index_bgen",
        TInt64,
        ArraySeq(tlocus("GRCh37")),
        Str(bgen),
        Str(idx),
        Literal(TDict(TString, TString), contigRecoding),
        False(),
        I32(1000000),
      )
    )
}

class BgenIndexSuite {
  import BgenIndexSuite._

  private val example = getTestResource("example.8bits.bgen")

  private def indexExample(idx: String)(implicit ctx: ExecuteContext): Unit =
    indexBgen(example, idx, Map("01" -> "1"))

  /** The example indexed once with each locus representation canonical, struct first. */
  private def indexExampleUnderEachRepresentation()(implicit ctx: ExecuteContext)
    : IndexedSeq[String] = {
    val idxs = ArraySeq.newBuilder[String]
    forEachCanonicalLocus { _ =>
      val idx = ctx.createTmpPath("test-bgen-index-repr", "idx2")
      indexExample(idx)
      idxs += idx
    }
    idxs.result()
  }

  private def readRows(idx: String, bgen: String = example)(implicit ctx: ExecuteContext): Any = {
    val reader = MatrixBGENReader(ctx, FastSeq(bgen), None, Map(bgen -> idx), Some(3), None, None)
    val rows = MatrixRowsTable(MatrixRead(reader.fullMatrixType, true, false, reader))
    loweredExecute(TableCollect(TableKeyBy(rows, FastSeq())), Env.empty, FastSeq(), None)
  }

  private def bytes(path: String)(implicit ctx: ExecuteContext): Array[Byte] =
    using(ctx.fs.open(path))(_.readAllBytes())

  // The BGEN index reader does not record encodings; it rebuilds them from the index version
  // (BgenSettings.getIndexSpec). index_bgen must write that encoding whatever the flags say.
  @Test def testIndexReadableRegardlessOfUnstableEncodings(implicit ctx: ExecuteContext): Unit = {
    val stableIdx = ctx.createTmpPath("test-bgen-index-stable", "idx2")
    val expected = withStableEncodings { implicit ctx =>
      indexExample(stableIdx)
      readRows(stableIdx)
    }

    val unstableIdx = ctx.createTmpPath("test-bgen-index-unstable", "idx2")
    withUnstableEncodings(implicit ctx => indexExample(unstableIdx))

    withStableEncodings(implicit ctx => assert(readRows(unstableIdx) == expected))
    withUnstableEncodings(implicit ctx => assert(readRows(unstableIdx) == expected))
  }

  // The bytes of a BGEN index come from its encoding alone, so they don't depend on which locus
  // representation is canonical when it is written.
  @Test def testIndexBytesIndependentOfLocusRepresentation(implicit ctx: ExecuteContext): Unit = {
    val ArraySeq(struct, packed) = indexExampleUnderEachRepresentation().map(bytes): @unchecked
    assert(struct.sameElements(packed))
  }

  // An index written under either representation reads under both, with the same rows.
  @Test def testIndexReadableAcrossLocusRepresentations(implicit ctx: ExecuteContext): Unit = {
    val expected = readRows(getTestResource("example.8bits.bgen.idx2"))
    for (idx <- indexExampleUnderEachRepresentation())
      forEachCanonicalLocus(_ => assertEq(readRows(idx), expected))
  }

  // Indexes checked in as test resources were written by earlier versions of Hail, before packed
  // loci existed: 1.1.0 (a directory) and 1.3.0 (a single file).
  @Test def testExistingIndexesReadUnderEitherLocusRepresentation(implicit ctx: ExecuteContext)
    : Unit =
    for (name <- FastSeq("random.bgen", "example.8bits.bgen")) {
      val path = getTestResource(name)
      val idx = getTestResource(s"$name.idx2")
      val expected = readRows(idx, path)
      forEachCanonicalLocus(_ => assertEq(readRows(idx, path), expected))
    }
}
