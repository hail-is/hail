package is.hail.variant

import java.util.ArrayList

import htsjdk.samtools.SAMSequenceDictionary
import is.hail.annotations.Annotation
import is.hail.expr
import is.hail.expr.{AnnotationImpex, SparkAnnotationImpex}
import org.apache.kudu.client.{CreateTableOptions, KuduClient}
import org.apache.kudu.{ColumnSchema, Schema, Type}
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._

import scala.collection.JavaConversions._
import scala.collection.mutable
import scala.collection.mutable.ListBuffer

object KuduAnnotationImpex extends AnnotationImpex[DataType, Any] {
  def supportsType(t: expr.Type): Boolean = true

  def exportType(t: expr.Type): DataType = flatten(SparkAnnotationImpex.exportType(t))

  def exportAnnotation(a: Annotation, t: expr.Type, gr: GenomeReference): Any = flatten(SparkAnnotationImpex.exportAnnotation(a, t, gr), t)

  def importAnnotation(a: Any, t: expr.Type, gr: GenomeReference): Annotation =
    SparkAnnotationImpex.importAnnotation(unflatten(a, t), t, gr)

  val fixedArraySize = 2

  def flatten(dt: DataType): DataType = dt match {
    case st: StructType => StructType(flattenFields(st, "", ListBuffer[StructField]()))
    case _ => dt
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

  def flatten(a: Any, t: expr.Type): Any = t match {
    case st: expr.TStruct => Row.fromSeq(flattenElements(a.asInstanceOf[Row], st, ListBuffer[Any]()))
    case _ => a
  }

  private def flattenElements(row: Row, t: expr.TStruct, acc: Seq[Any]): Seq[Any] = {
    t.fields.flatMap(field => field.typ match {
      case st: expr.TStruct => flattenElements(row.getStruct(field.index), st, acc)
      case at: expr.TIterable => at.elementType match {
        case st: expr.TStruct => row.getAs[Seq[Row]](field.index)
          .take(fixedArraySize).padTo(fixedArraySize, null)
          .flatMap(a => flattenElements(a, st, acc))
        case _ => row.getSeq(field.index)
          .take(fixedArraySize).padTo(fixedArraySize, null)
      }
      case mt: expr.TDict =>
        throw new IllegalArgumentException("Cannot flatten MapType")
      case _ => if (row == null) Some(null) else Some(row.get(field.index))
    })
  }

  def unflatten(a: Annotation, t: expr.Type): Any = (t: @unchecked) match {
    case st: expr.TStruct =>
      val row = a.asInstanceOf[Row]
      val it = row.toSeq.iterator
      Row.fromSeq(st.fields.map(f => consume(it, f.typ)))
  }

  def reorder(row: Row, indices: Array[Int]): Row = {
    val values: Seq[Any] = row.toSeq
    Row.fromSeq(for (i <- List.range(0, values.length)) yield values(indices(i)))
  }

  private def consume(stream: Iterator[Any], t: expr.Type): Any = {
    t match {
      case st: expr.TStruct =>
        val values = st.fields.map(f => consume(stream, f.typ))
        if (values.forall(_ == null)) null else Row.fromSeq(values)
      case at: expr.TIterable => mutable.WrappedArray.make(
        Range(0, fixedArraySize).toArray
          .map(f => consume(stream, at.elementType)).filter(_ != null))
      case _ => stream.next()
    }
  }
}

object KuduUtils {

  def toKuduSchema(schema: StructType, keys: Seq[String]): Schema = {
    val kuduCols = new ArrayList[ColumnSchema]()
    // add the key columns first, in the order specified
    for (key <- keys) {
      val f = schema.fields(schema.fieldIndex(key))
      kuduCols.add(new ColumnSchema.ColumnSchemaBuilder(f.name, kuduType(f.dataType)).key(true).build())
    }
    // now add the non-key columns
    for (f <- schema.fields.filter(field => !keys.contains(field.name))) {
      kuduCols.add(new ColumnSchema.ColumnSchemaBuilder(f.name, kuduType(f.dataType)).nullable(f.nullable).key(false).build())
    }
    new Schema(kuduCols)
  }

  def kuduType(dt: DataType): Type = dt match {
    case DataTypes.BinaryType => Type.BINARY
    case DataTypes.BooleanType => Type.BOOL
    case DataTypes.StringType => Type.STRING
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
    seqDict.getSequences.flatMap { seq =>
      val contig = seq.getSequenceName
      val length = seq.getSequenceLength
      Range(0, 1 + length / rowsPerPartition).map(pos => (contig, pos * rowsPerPartition))
    }.map { case (contig, start) =>
      val partialRow = toKuduSchema(schema, keys).newPartialRow()
      partialRow.addString("variant__contig", contig)
      partialRow.addInt("variant__start", start)
      options.addSplitRow(partialRow)
    }
    options
  }

  def tableExists(masterAddress: String, tableName: String): Boolean = {
    val client = new KuduClient.KuduClientBuilder(masterAddress).build
    try {
      client.tableExists(tableName)
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

}
