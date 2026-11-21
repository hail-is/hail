package is.hail.io.columnar

import is.hail.TestUtils._
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.types.virtual._
import is.hail.utils.Interval

import scala.jdk.CollectionConverters._

import java.util.regex.Pattern

import org.apache.arrow.vector.types.pojo.{ArrowType, Field, FieldType, Schema}
import org.apache.spark.sql.Row
import org.junit.jupiter.api.Test

class ColumnarPartitionsTableSuite {

  private def fixture(name: String): String = getTestResource(s"columnar/0.1/table/$name")

  private def partitionsSchema(name: String)(implicit ctx: ExecuteContext): Schema =
    ColumnarFileSource.readArrowSchema(ctx.fs, s"${fixture(name)}/partitions.parquet")

  private def rowsSchema(name: String)(implicit ctx: ExecuteContext): Schema =
    ColumnarManifest.read(ctx.fs, fixture(name)).rowsSchema

  private val mtFixture: String = getTestResource("columnar/0.1/matrix_table/missing_entries")

  private def mtPartitionsSchema(implicit ctx: ExecuteContext): Schema =
    ColumnarFileSource.readArrowSchema(ctx.fs, s"$mtFixture/partitions.parquet")

  private def mtRowsSchema(implicit ctx: ExecuteContext): Schema =
    ColumnarManifest.read(ctx.fs, mtFixture).rowsSchema

  private def withFields(schema: Schema)(f: Seq[Field] => Seq[Field]): Schema =
    new Schema(f(schema.getFields.asScala.toSeq).asJava)

  private def assertRejected(
    partitions: Schema,
    rows: Schema,
    key: IndexedSeq[String],
    message: String,
    matrix: Boolean = false,
  ): Unit =
    interceptFatal(Pattern.quote(message)) {
      ColumnarPartitionsTable.checkSchema(partitions, rows, key, matrix)
    }

  @Test def testAcceptsTheFixturesPartitionsTables(implicit ctx: ExecuteContext): Unit = {
    ColumnarPartitionsTable.checkSchema(
      partitionsSchema("keyed"),
      rowsSchema("keyed"),
      FastSeq("idx"),
    )
    ColumnarPartitionsTable.checkSchema(
      partitionsSchema("unkeyed"),
      rowsSchema("unkeyed"),
      FastSeq(),
    )
  }

  @Test def testRejectsBoundsInAnUnkeyedTable(implicit ctx: ExecuteContext): Unit =
    assertRejected(partitionsSchema("keyed"), rowsSchema("keyed"), FastSeq(), "field 'start'")

  @Test def testRejectsMissingBoundsInAKeyedTable(implicit ctx: ExecuteContext): Unit =
    assertRejected(
      partitionsSchema("unkeyed"),
      rowsSchema("unkeyed"),
      FastSeq("idx"),
      "missing field 'start'",
    )

  @Test def testRejectsEntryFilesInATable(implicit ctx: ExecuteContext): Unit = {
    val entryFiles = new Field(
      "entry_files",
      new FieldType(false, ArrowType.List.INSTANCE, null),
      FastSeq(partitionsSchema("keyed").findField("row_file")).asJava,
    )
    assertRejected(
      withFields(partitionsSchema("keyed"))(_ :+ entryFiles),
      rowsSchema("keyed"),
      FastSeq("idx"),
      "field 'entry_files'",
    )
  }

  @Test def testAcceptsAMatrixTablesPartitionsTable(implicit ctx: ExecuteContext): Unit =
    ColumnarPartitionsTable.checkSchema(
      mtPartitionsSchema,
      mtRowsSchema,
      FastSeq("row_idx"),
      matrix = true,
    )

  @Test def testRejectsAMatrixTableWithoutEntryFiles(implicit ctx: ExecuteContext): Unit =
    assertRejected(
      partitionsSchema("keyed"),
      rowsSchema("keyed"),
      FastSeq("idx"),
      "missing field 'entry_files'",
      matrix = true,
    )

  @Test def testRejectsMistypedEntryFiles(implicit ctx: ExecuteContext): Unit = {
    // A single file reference, where the spec has a list of them.
    val partitions = withFields(mtPartitionsSchema)(_.map { f =>
      if (f.getName == "entry_files") {
        val rowFile = mtPartitionsSchema.findField("row_file")
        new Field("entry_files", rowFile.getFieldType, rowFile.getChildren)
      } else f
    })
    assertRejected(
      partitions,
      mtRowsSchema,
      FastSeq("row_idx"),
      "field 'entry_files' has the wrong type",
      matrix = true,
    )
  }

  @Test def testRejectsNullableEntryFiles(implicit ctx: ExecuteContext): Unit = {
    val partitions = withFields(mtPartitionsSchema)(_.map { f =>
      if (f.getName == "entry_files")
        new Field("entry_files", FieldType.nullable(f.getType), f.getChildren)
      else f
    })
    assertRejected(
      partitions,
      mtRowsSchema,
      FastSeq("row_idx"),
      "field 'entry_files' has the wrong type",
      matrix = true,
    )
  }

  @Test def testRejectsUnknownFields(implicit ctx: ExecuteContext): Unit = {
    val extra = new Field("extra", FieldType.nullable(new ArrowType.Int(32, true)), null)
    assertRejected(
      withFields(partitionsSchema("keyed"))(_ :+ extra),
      rowsSchema("keyed"),
      FastSeq("idx"),
      "field 'extra'",
    )
  }

  @Test def testRejectsBoundsOfAnotherTypeThanTheKey(implicit ctx: ExecuteContext): Unit = {
    // The `s` field of the keyed fixture is a string, and the bounds hold `idx`, an int32.
    val rows = withFields(rowsSchema("keyed"))(_.map { f =>
      if (f.getName == "s") new Field("idx", f.getFieldType, f.getChildren)
      else if (f.getName == "idx") new Field("s", f.getFieldType, f.getChildren)
      else f
    })
    assertRejected(partitionsSchema("keyed"), rows, FastSeq("idx"), "field 'start'")
  }

  @Test def testRejectsAMistypedRowCount(implicit ctx: ExecuteContext): Unit = {
    val partitions = withFields(partitionsSchema("keyed"))(_.map { f =>
      if (f.getName == "row_count")
        new Field("row_count", FieldType.notNullable(new ArrowType.Int(32, true)), null)
      else f
    })
    assertRejected(partitions, rowsSchema("keyed"), FastSeq("idx"), "field 'row_count'")
  }

  @Test def testKeyedPartitionerComesFromTheBounds(implicit ctx: ExecuteContext): Unit = {
    val reader = TableColumnarReader(ctx.fs, ctx.references, fixture("keyed"))
    val stage = reader.lower(ctx, reader.fullType)
    assertEq(stage.partitioner.kType, TStruct("idx" -> TInt32))
    assertEq(
      stage.partitioner.rangeBounds,
      FastSeq(
        Interval(Row(0), Row(2), includesStart = true, includesEnd = true),
        Interval(Row(3), Row(5), includesStart = true, includesEnd = true),
      ),
    )
    assertEq(stage.key, FastSeq("idx"))
    assert(reader.isDistinctlyKeyed)
  }

  @Test def testUnkeyedPartitionerHasOnePartitionPerRowFile(implicit ctx: ExecuteContext): Unit = {
    val reader = TableColumnarReader(ctx.fs, ctx.references, fixture("unkeyed"))
    val stage = reader.lower(ctx, reader.fullType)
    assertEq(stage.partitioner.kType, TStruct.empty)
    assertEq(stage.numPartitions, 2)
    assert(!reader.isDistinctlyKeyed)
  }
}
