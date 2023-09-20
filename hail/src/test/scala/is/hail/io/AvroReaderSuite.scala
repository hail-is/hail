package is.hail.io

import is.hail.ExecStrategy.ExecStrategy
import is.hail.expr.ir.{I64, MakeStruct, ReadPartition, Str, ToArray}
import is.hail.io.avro.AvroPartitionReader
import is.hail.utils.{FastSeq, fatal, using}
import is.hail.{ExecStrategy, HailSuite}
import org.apache.avro.SchemaBuilder
import org.apache.avro.file.DataFileWriter
import org.apache.avro.generic.{GenericDatumWriter, GenericRecord, GenericRecordBuilder}
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class AvroReaderSuite extends HailSuite {
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
    Row(0, null, 0f, 0d, null),
    Row(1, 1L, 1.0f, 1.0d, ""),
    Row(-1, -1L, -1.0f, -1.0d, "minus one"),
    Row(Int.MaxValue, Long.MaxValue, Float.MaxValue, Double.MaxValue, null),
    Row(Int.MinValue, null, Float.MinPositiveValue, Double.MinPositiveValue, "MINIMUM STRING")
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

  def makeTestFile(): String = {
    val avroFile = ctx.createTmpPath("avro_test", "avro")

    using(fs.create(avroFile)) { os =>
      using(new DataFileWriter[GenericRecord](new GenericDatumWriter(testSchema)).create(testSchema, os)) { dw =>
        for (row <- testValue) {
          dw.append(makeRecord(row))
        }
      }
    }

    avroFile
  }

  @Test def avroReaderWorks(): Unit = {
    val avroFile = makeTestFile()
    val ir = ToArray(ReadPartition(
      MakeStruct(Array("partitionPath" -> Str(avroFile), "partitionIndex" -> I64(0))),
      partitionReader.fullRowType,
      partitionReader))
    val testValueWithUIDs = testValue.zipWithIndex.map { case(x, i) =>
      Row(x(0), x(1), x(2), x(3), x(4), Row(0L, i.toLong))
    }
    assertEvalsTo(ir, testValueWithUIDs)
  }

  @Test def testSmallerRequestedType(): Unit = {
    val avroFile = makeTestFile()
    val ir = ToArray(ReadPartition(
      MakeStruct(Array("partitionPath" -> Str(avroFile), "partitionIndex" -> I64(0))),
      partitionReader.fullRowType.typeAfterSelect(FastSeq(0, 2, 4)),
      partitionReader))
    val expected = testValue.map { case Row(int, _, float, _, string) => Row(int, float, string) }
    assertEvalsTo(ir, expected)
  }
}
