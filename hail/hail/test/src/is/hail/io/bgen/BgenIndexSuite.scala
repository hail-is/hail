package is.hail.io.bgen

import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir.{invoke, Env, MatrixRead, MatrixRowsTable, TableKeyBy}
import is.hail.expr.ir.defs.{False, I32, Literal, Str, TableCollect}
import is.hail.types.virtual._

import scala.collection.immutable.ArraySeq

import org.junit.jupiter.api.Test

class BgenIndexSuite {
  private val bgen = getTestResource("example.8bits.bgen")

  private def indexBgen(idx: String)(implicit ctx: ExecuteContext): Unit =
    eval(
      invoke(
        "index_bgen",
        TInt64,
        ArraySeq(TLocus("GRCh37")),
        Str(bgen),
        Str(idx),
        Literal(TDict(TString, TString), Map("01" -> "1")),
        False(),
        I32(1000000),
      )
    )

  private def readRows(idx: String)(implicit ctx: ExecuteContext): Any = {
    val reader = MatrixBGENReader(ctx, FastSeq(bgen), None, Map(bgen -> idx), Some(3), None, None)
    val rows = MatrixRowsTable(MatrixRead(reader.fullMatrixType, true, false, reader))
    loweredExecute(TableCollect(TableKeyBy(rows, FastSeq())), Env.empty, FastSeq(), None)
  }

  // The BGEN index reader does not record encodings; it rebuilds them from the index version
  // (BgenSettings.getIndexSpec). index_bgen must write that encoding whatever the flags say.
  @Test def testIndexReadableRegardlessOfUnstableEncodings(implicit ctx: ExecuteContext): Unit = {
    val stableIdx = ctx.createTmpPath("test-bgen-index-stable", "idx2")
    indexBgen(stableIdx)
    val expected = readRows(stableIdx)

    val unstableIdx = ctx.createTmpPath("test-bgen-index-unstable", "idx2")
    withUnstableEncodings(implicit ctx => indexBgen(unstableIdx))

    assert(readRows(unstableIdx) == expected)
    withUnstableEncodings(implicit ctx => assert(readRows(unstableIdx) == expected))
  }
}
