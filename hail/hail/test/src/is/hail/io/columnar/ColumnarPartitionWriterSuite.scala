package is.hail.io.columnar

import is.hail.ExecStrategy
import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.ir.defs.{I32, MakeStruct, NA, Str, WritePartition}
import is.hail.types.physical.{PCanonicalStruct, PInt32}
import is.hail.types.virtual.{TInt32, TStream, TStruct}

import org.junit.jupiter.api.Test

class ColumnarPartitionWriterSuite {

  // Emit makes a write missing when its stream is missing, so the requiredness analysis must too.
  @Test def testWriteOfMissingStreamIsMissing(implicit ctx: ExecuteContext): Unit = {
    implicit val execStrats = ExecStrategy.compileOnly
    val writer = ColumnarPartitionWriter(
      ctx.createTmpPath("columnar-partition-writer"),
      PCanonicalStruct(true, "x" -> PInt32(true)),
      FastSeq(),
      None,
    )
    val write = WritePartition(
      NA(TStream(TStruct("x" -> TInt32))),
      MakeStruct(FastSeq("path" -> Str("part.parquet"), "partitionIndex" -> I32(0))),
      writer,
    )
    assertEvalsTo(write, null)
  }
}
