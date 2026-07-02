package is.hail.io

import is.hail.ExecStrategy
import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils._
import is.hail.annotations.RowSeq
import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.collection.compat.immutable.ArraySeq
import is.hail.expr.ir.defs.{I64, MakeStruct, ReadPartition, Str, ToArray}
import is.hail.io.avro.AvroPartitionReader
import is.hail.utils.{fatal, using}

import org.apache.avro.SchemaBuilder
import org.apache.avro.file.DataFileWriter
import org.apache.avro.generic.{GenericDatumWriter, GenericRecord, GenericRecordBuilder}
import org.apache.spark.sql.Row
import org.junit.jupiter.api.Test

class AvroReaderSuite {
  implicit val execStrats: Set[ExecStrategy] = Set(ExecStrategy.LoweredJVMCompile)

  private val testSchema = SchemaBuilder.record("Root")
    .fields()
    .name("an_int").`type`().intType().noDefault()
    .name("an_optional_long").`type`().nullable().longType().noDefault()
    .name("a_float").`type`().floatType().noDefault()
    .name("a_double").`type`().doubleType().noDefault()
    .name("an_optional_string").`type`().nullable().stringType().noDefault()
    .endRecord()

  private val testValue = IndexedSeq(
    RowSeq(0, null, 0f, 0d, null),
    RowSeq(1, 1L, 1.0f, 1.0d, ""),
    RowSeq(-1, -1L, -1.0f, -1.0d, "minus one"),
    RowSeq(Int.MaxValue, Long.MaxValue, Float.MaxValue, Double.MaxValue, null),
    RowSeq(Int.MinValue, null, Float.MinPositiveValue, Double.MinPositiveValue, "MINIMUM STRING"),
  )

  private val partitionReader = AvroPartitionReader(testSchema, "rowUID")

  def makeRecord(row: Row): GenericRecord = row match {
    case Row(int, long, float, double, string) => new GenericRecordBuilder(testSchema)
        .set("an_int", int)
        .set("an_optional_long", long)
        .set("a_float", float)
        .set("a_double", double)
        .set("an_optional_string", string)
        .build()
    case _ => fatal("invalid row")
  }

  def makeTestFile()(implicit ctx: ExecuteContext): String = {
    val avroFile = ctx.createTmpPath("avro_test", "avro")

    using(ctx.fs.create(avroFile)) { os =>
      using(new DataFileWriter[GenericRecord](new GenericDatumWriter(testSchema)).create(
        testSchema,
        os,
      )) { dw =>
        for (row <- testValue)
          dw.append(makeRecord(row))
      }
    }

    avroFile
  }

  @Test def avroReaderWorks(implicit ctx: ExecuteContext): Unit = {
    val avroFile = makeTestFile()
    val ir = ToArray(ReadPartition(
      MakeStruct(ArraySeq("partitionPath" -> Str(avroFile), "partitionIndex" -> I64(0))),
      partitionReader.fullRowType,
      partitionReader,
    ))
    val testValueWithUIDs = testValue.zipWithIndex.map { case (x, i) =>
      RowSeq(x(0), x(1), x(2), x(3), x(4), RowSeq(0L, i.toLong))
    }
    assertEvalsTo(ir, testValueWithUIDs)
  }

  @Test def testSmallerRequestedType(implicit ctx: ExecuteContext): Unit = {
    val avroFile = makeTestFile()
    val ir = ToArray(ReadPartition(
      MakeStruct(ArraySeq("partitionPath" -> Str(avroFile), "partitionIndex" -> I64(0))),
      partitionReader.fullRowType.typeAfterSelect(FastSeq(0, 2, 4)),
      partitionReader,
    ))
    val expected = testValue.map { case Row(int, _, float, _, string) =>
      RowSeq(int, float, string)
    }
    assertEvalsTo(ir, expected)
  }
}
