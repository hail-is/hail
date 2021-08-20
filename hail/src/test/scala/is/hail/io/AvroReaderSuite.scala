package is.hail.io

import is.hail.ExecStrategy.ExecStrategy
import is.hail.TestUtils.assertEvalsTo
import is.hail.expr.ir.{ReadPartition, Str, ToArray}
import is.hail.io.avro.AvroPartitionReader
import is.hail.{ExecStrategy, HailSuite}
import org.apache.avro.SchemaBuilder
import org.apache.avro.file.DataFileWriter
import org.apache.avro.generic.{GenericDatumWriter, GenericRecord, GenericRecordBuilder}
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class AvroReaderSuite extends HailSuite {
  implicit val execStrats: Set[ExecStrategy] = Set(ExecStrategy.Interpret, ExecStrategy.InterpretUnoptimized, ExecStrategy.LoweredJVMCompile)

  @Test def avroReaderWorks(): Unit = {
    val avroFile = ctx.createTmpPath("avro_test", "avro")
    val schema = SchemaBuilder.record("Root")
      .fields()
      .name("an_int").`type`().intType().noDefault()
      .name("an_optional_long").`type`().nullable().longType().noDefault()
      .name("a_float").`type`().floatType().noDefault()
      .name("a_double").`type`().doubleType().noDefault()
      .name("an_optional_string").`type`().nullable().stringType().noDefault()
      .endRecord()
    val testValue = Array(
      Row(0, null, 0f, 0d, null),
      Row(1, 1L, 1.0f, 1.0d, ""),
      Row(-1, -1L, -1.0f, -1.0d, "minus one"),
      Row(Int.MaxValue, Long.MaxValue, Float.MaxValue, Double.MaxValue, null),
      Row(Int.MinValue, null, Float.MinPositiveValue, Double.MinPositiveValue, "MINIMUM STRING")
    )

    val os = fs.create(avroFile)
    val dw = new DataFileWriter[GenericRecord](new GenericDatumWriter(schema)).create(schema, os)
    for (Row(i: Int, l: Long, f: Float, d: Double, s: String) <- testValue) {
      val gr = new GenericRecordBuilder(schema)
        .set("an_int", i)
        .set("an_optional_long", l)
        .set("a_float", f)
        .set("a_double", d)
        .set("an_optional_string", s)
        .build()

      dw.append(gr)
    }
    dw.close()

    val partitionReader = new AvroPartitionReader(schema)
    val ir = ToArray(ReadPartition(Str(avroFile), partitionReader.fullRowType, partitionReader))
    assertEvalsTo(ir, testValue)
  }
}
