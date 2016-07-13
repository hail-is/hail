package org.broadinstitute.hail.variant

import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.broadinstitute.hail.expr._
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class KuduUtilsSuite extends TestNGSuite {
  @Test def testFlattenPrimitives() {
    val schema = StructType(List(
      StructField("f1", IntegerType), StructField("f2", StringType)))
    assert(KuduAnnotationImpex.flatten(schema) == schema)

    val row = Row(1, "a")
    roundTrip(schema, row, row)
  }

  @Test def testFlattenArray() {
    val schema = StructType(List(StructField("f1", ArrayType(IntegerType))))
    val flattenedSchema = StructType(List(
      StructField("f1_0", IntegerType), StructField("f1_1", IntegerType)))
    assert(KuduAnnotationImpex.flatten(schema) == flattenedSchema)

    roundTrip(schema, Row(List()), Row(null, null))
    roundTrip(schema, Row(List(1)), Row(1, null))
    roundTrip(schema, Row(List(1, 2)), Row(1, 2))
    assert(KuduAnnotationImpex.flatten(Row(List(1, 2, 3)),
      SparkAnnotationImpex.importType(schema)) == Row(1, 2))
  }

  @Test def testFlattenNested() {
    val schema = StructType(List(
      StructField("f1", StructType(List(StructField("s1", IntegerType))))))
    val flattenedSchema = StructType(List(StructField("f1__s1", IntegerType)))
    assert(KuduAnnotationImpex.flatten(schema) == flattenedSchema)

    roundTrip(schema, Row(Row(1)), Row(1))
  }

  @Test def testFlattenDeepNested() {
    val schema = StructType(List(
      StructField("f1", LongType),
      StructField("f2", StructType(List(
        StructField("s1", StructType(List(StructField("t1", IntegerType))))))),
      StructField("f3", LongType)
    ))
    val flattenedSchema = StructType(List(
      StructField("f1", LongType),
      StructField("f2__s1__t1", IntegerType),
      StructField("f3", LongType)))
    assert(KuduAnnotationImpex.flatten(schema) == flattenedSchema)

    roundTrip(schema, Row(1L, Row(Row(2)), 3L), Row(1L, 2, 3L))
  }

  @Test def testFlattenArrayOfStruct() {
    val schema = StructType(List(
      StructField("f1", ArrayType(StructType(List(
        StructField("s1", IntegerType),
        StructField("s2", StringType)))))
    ))
    val flattenedSchema = StructType(List(
      StructField("f1_0__s1", IntegerType),
      StructField("f1_0__s2", StringType),
      StructField("f1_1__s1", IntegerType),
      StructField("f1_1__s2", StringType)))
    assert(KuduAnnotationImpex.flatten(schema) == flattenedSchema)

    roundTrip(schema, Row(List(Row(1, "a"))), Row(1, "a", null, null))
    roundTrip(schema, Row(List(Row(1, "a"), Row(2, "b"))), Row(1, "a", 2, "b"))
  }

  @Test def testFlattenComplex() {
    val schema = StructType(List(
      StructField("f1", ArrayType(LongType)),
      StructField("f2", StructType(List(
        StructField("s1", BooleanType),
        // StructField("s2", ByteType),
        // StructField("s3", ShortType),
        StructField("s4", IntegerType),
        StructField("s5", LongType),
        StructField("s6", FloatType),
        StructField("s7", DoubleType),
        StructField("s8", StringType),
        StructField("s9", BinaryType))))))
    val flattenedSchema = StructType(List(
      StructField("f1_0", LongType),
      StructField("f1_1", LongType),
      StructField("f2__s1", BooleanType),
      // StructField("f2__s2", ByteType),
      // StructField("f2__s3", ShortType),
      StructField("f2__s4", IntegerType),
      StructField("f2__s5", LongType),
      StructField("f2__s6", FloatType),
      StructField("f2__s7", DoubleType),
      StructField("f2__s8", StringType),
      StructField("f2__s9", BinaryType)))
    assert(KuduAnnotationImpex.flatten(schema) == flattenedSchema)

    roundTrip(schema, Row(List(1L, 2L), Row(true,
      // 3.toByte, 4.toShort,
      5, 6L, 7.toFloat,
      8.0, "a", Array(10.toByte, 11.toByte))),
      Row(1L, 2L, true,
        // 3.toByte, 4.toShort,
        5, 6L, 7.toFloat, 8.0, "a",
        Array(10.toByte, 11.toByte)))
  }

  def roundTrip(schema: StructType, row: Row, flattenedRow: Row): Unit = {
    assert(KuduAnnotationImpex.flatten(row, SparkAnnotationImpex.importType(schema)) == flattenedRow)
    assert(KuduAnnotationImpex.unflatten(flattenedRow, SparkAnnotationImpex.importType(schema)) == row)
  }

}
