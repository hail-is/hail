package org.broadinstitute.hail.variant

import java.util.ArrayList

import htsjdk.samtools.SAMSequenceDictionary
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.kududb.ColumnSchema.ColumnSchemaBuilder
import org.kududb.{ColumnSchema, Schema, Type}
import org.kududb.client.{CreateTableOptions, KuduClient}

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

object KuduUtils {

  def toKuduSchema(schema: StructType, keys: Seq[String]): Schema = {
    val kuduCols = new ArrayList[ColumnSchema]()
    // add the key columns first, in the order specified
    for (key <- keys) {
      val f = schema.fields(schema.fieldIndex(key))
      kuduCols.add(new ColumnSchema.ColumnSchemaBuilder(f.name, kuduType(f.dataType)).key(true).build())
    }
    // now add the non-key columns
    for (f <- schema.fields.filter(field=> !keys.contains(field.name))) {
      kuduCols.add(new ColumnSchema.ColumnSchemaBuilder(f.name, kuduType(f.dataType)).nullable(f.nullable).key(false).build())
    }
    new Schema(kuduCols)
  }

  def kuduType(dt: DataType) : Type = dt match {
    case DataTypes.BinaryType => Type.BINARY
    case DataTypes.BooleanType => Type.BOOL
    case DataTypes.StringType => Type.STRING
    case DataTypes.TimestampType => Type.TIMESTAMP
    case DataTypes.ByteType => Type.INT8
    case DataTypes.ShortType => Type.INT16
    case DataTypes.IntegerType => Type.INT32
    case DataTypes.LongType => Type.INT64
    case DataTypes.FloatType => Type.FLOAT
    case DataTypes.DoubleType => Type.DOUBLE
    case _ => throw new IllegalArgumentException(s"No support for Spark SQL type $dt")
  }

  def createTableOptions(schema: StructType, keys: Seq[String], seqDict: SAMSequenceDictionary, rowsPerPartition: Int): CreateTableOptions = {
    val rangePartitionCols = new ArrayList[String]
    rangePartitionCols.add("variant__contig")
    rangePartitionCols.add("variant__start")
    val options = new CreateTableOptions().setRangePartitionColumns(rangePartitionCols)
    // e.g. if rowsPerPartition is 10000000 then create split points at
    // (' 1', 0), (' 1', 10000000), (' 1', 20000000), ... (' 1', 240000000)
    // (' 2', 0), (' 2', 10000000), (' 2', 20000000), ... (' 2', 240000000)
    // ...
    // (' Y', 0), (' Y', 10000000), (' Y', 20000000), ... (' Y', 50000000)
    seqDict.getSequences.flatMap(seq => {
      val contig = seq.getSequenceName
      val length = seq.getSequenceLength
      Range(0, 1 + length/rowsPerPartition).toList.map(pos => (contig, pos*rowsPerPartition))
    }).map(kv => {
      val partialRow = toKuduSchema(schema, keys).newPartialRow()
      partialRow.addString("variant__contig", kv._1)
      partialRow.addInt("variant__start", kv._2)
      options.addSplitRow(partialRow)
    })
    options
  }

  def tableExists(masterAddress: String, tableName: String): Boolean = {
    val client = new KuduClient.KuduClientBuilder(masterAddress).build
    try {
      return client.tableExists(tableName)
    } finally {
      client.close()
    }
  }

  def dropTable(masterAddress: String, tableName: String) {
    val client = new KuduClient.KuduClientBuilder(masterAddress).build
    try {
      if (client.tableExists(tableName)) {
        println("Dropping table " + tableName)
        client.deleteTable(tableName)
      }
    } finally {
      client.close()
    }
  }

  val fixedArraySize = 2

  def flatten(schema: StructType): StructType = {
    StructType(flattenFields(schema, "", ListBuffer[StructField]()))
  }

  private def flattenFields(schema: StructType, prefix: String, acc: Seq[StructField]): Seq[StructField] = {
    schema.flatMap(field => field.dataType match {
      case st: StructType => flattenFields(st, prefix + field.name + "__", acc)
      case at: ArrayType => at.elementType match {
        case st: StructType => Range(0, fixedArraySize).flatMap(a =>
          flattenFields(makeNullable(st), prefix + field.name + "_" + a + "__", acc)
        )
        case _ => Range(0, fixedArraySize).map(a =>
          StructField(prefix + field.name + "_" + a, at.elementType, nullable = true, field.metadata)
        )
      }
      case mt: MapType =>
        throw new IllegalArgumentException("Cannot flatten MapType")
      case _ => Some(field.copy(name = prefix + field.name))
    })
  }

  private def makeNullable(st: StructType): StructType = {
    st.copy(st.fields.map(f => f.copy(nullable = true)))
  }

  def flatten(row: Row, schema: StructType): Row = {
    Row.fromSeq(flattenElements(row, schema, ListBuffer[Any]()))
  }

  private def flattenElements(row: Row, schema: StructType, acc: Seq[Any]): Seq[Any] = {
    schema.flatMap(field => field.dataType match {
      case st: StructType => flattenElements(row.getStruct(schema.fieldIndex(field
        .name)), st, acc)
      case at: ArrayType => at.elementType match {
        case st: StructType => row.getAs[Seq[Row]](schema.fieldIndex(field.name))
          .take(fixedArraySize).padTo(fixedArraySize, null)
          .flatMap(a => flattenElements(a, st, acc))
        case _ => row.getSeq(schema.fieldIndex(field.name))
          .take(fixedArraySize).padTo(fixedArraySize, null)
      }
      case mt: MapType =>
        throw new IllegalArgumentException("Cannot flatten MapType")
      case _ => if (row == null) Some(null) else Some(row.get(schema.fieldIndex(field.name)))
    })
  }

  def unflatten(row: Row, schema: StructType): Row = {
    val it = row.toSeq.iterator
    Row.fromSeq(schema.fields.map(f => consume(it, f.dataType)))
  }

  def reorder(row: Row, indices: Array[Int]): Row = {
    val values: Seq[Any] = row.toSeq
    Row.fromSeq(for (i <- List.range(0, values.length)) yield values(indices(i)))
  }

  private def consume(stream: Iterator[Any], dataType: DataType): Any = {
    dataType match {
      case st: StructType => {
        val values = st.fields.map(f => consume(stream, f.dataType))
        if (values.forall(_ == null)) null else Row.fromSeq(values)
      }
      case at: ArrayType => mutable.WrappedArray.make(
        Range(0, fixedArraySize).toArray
        .map(f => consume(stream, at.elementType)).filter(_ != null))
      case _ => stream.next()
    }
  }

}
