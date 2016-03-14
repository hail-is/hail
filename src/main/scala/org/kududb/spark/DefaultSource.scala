package org.kududb.spark

import org.apache.spark.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.sources._
import org.apache.spark.sql.types._
import org.kududb.client.{RowResult}
import org.kududb.{Schema, ColumnSchema, Type}

import scala.collection.mutable

/**
 * DefaultSource for integration with Spark's dataframe datasources.
 * This class with produce a relationProvider based on input give to it from spark
 *
 * In all this DefaultSource support the following datasource functionality
 * - Scan range pruning through filter push down logic based on rowKeys
 * - Filter push down logic on columns that are not rowKey columns
 * - Qualifier filtering based on columns used in the SparkSQL statement
 * - Type conversions of basic SQL types
 */
class DefaultSource extends RelationProvider {

  val TABLE_KEY:String = "kudu.table"
  val KUDU_MASTER:String = "kudu.master"

  /**
   * Is given input from SparkSQL to construct a BaseRelation
   * @param sqlContext SparkSQL context
   * @param parameters Parameters given to us from SparkSQL
   * @return           A BaseRelation Object
   */
  override def createRelation(sqlContext: SQLContext,
                              parameters: Map[String, String]):
  BaseRelation = {


    val tableName = parameters.get(TABLE_KEY)
    if (tableName.isEmpty)
      new Throwable("Invalid value for " + TABLE_KEY +" '" + tableName + "'")

    val kuduMaster = parameters.getOrElse(KUDU_MASTER, "")

    new KuduRelation(tableName.get, kuduMaster)(sqlContext)
  }
}

/**
 * Implementation of Spark BaseRelation that will build up our scan logic
 * , do the scan pruning, filter push down, and value conversions
 *
 * @param tableName               Kudu table that we plan to read from
 * @param kuduMaster              Kudu master definition
 * @param sqlContext              SparkSQL context
 */
class KuduRelation (val tableName:String,
                     val kuduMaster: String) (
  @transient val sqlContext:SQLContext)
  extends BaseRelation with PrunedFilteredScan with Logging with Serializable {

  //create or get latest HBaseContext
  @transient var kuduContext = new KuduContext(sqlContext.sparkContext, kuduMaster)
  @transient var kuduClient = KuduClientCache.getKuduClient(kuduMaster)
  @transient var kuduTable = kuduClient.openTable(tableName)
  @transient var kuduSchema = kuduTable.getSchema
  @transient var kuduSchemaColumnMap = buildKuduSchemaColumnMap(kuduSchema)

  def getKuduSchemaColumnMap(): mutable.HashMap[String, ColumnSchema] = {
    if (kuduSchemaColumnMap == null) {
      kuduClient = KuduClientCache.getKuduClient(kuduMaster)
      kuduTable = kuduClient.openTable(tableName)
      kuduSchema = kuduTable.getSchema
      kuduSchemaColumnMap = buildKuduSchemaColumnMap(kuduSchema)
    }
    kuduSchemaColumnMap
  }

  def buildKuduSchemaColumnMap(kuduSchema:Schema): mutable.HashMap[String, ColumnSchema] = {

    var kuduSchemaColumnMap = new mutable.HashMap[String, ColumnSchema]()

    val columnIt = kuduSchema.getColumns.iterator()
    while (columnIt.hasNext) {
      val c = columnIt.next()
      kuduSchemaColumnMap.+=((c.getName, c))
    }
    kuduSchemaColumnMap
  }

  /**
   * Generates a Spark SQL schema object so Spark SQL knows what is being
   * provided by this BaseRelation
   *
   * @return schema generated from the SCHEMA_COLUMNS_MAPPING_KEY value
   */
  override def schema: StructType = {

    val metadataBuilder = new MetadataBuilder()

    val structFieldArray = new Array[StructField](kuduSchema.getColumnCount)

    val columnIt = kuduSchema.getColumns.iterator()
    var indexCounter = 0
    while (columnIt.hasNext) {
      val c = columnIt.next()

      val columnSparkSqlType = if (c.getType.equals(Type.BOOL)) BooleanType
      else if (c.getType.equals(Type.INT16)) IntegerType
      else if (c.getType.equals(Type.INT32)) IntegerType
      else if (c.getType.equals(Type.INT64)) LongType
      else if (c.getType.equals(Type.FLOAT)) FloatType
      else if (c.getType.equals(Type.DOUBLE)) DoubleType
      else if (c.getType.equals(Type.STRING)) StringType
      else if (c.getType.equals(Type.TIMESTAMP)) TimestampType
      else if (c.getType.equals(Type.BINARY)) BinaryType
      else throw new Throwable("Unsupported column type :" + c.getType)

      val metadata = metadataBuilder.putString("name", c.getName).build()
      val struckField =
        new StructField(c.getName, columnSparkSqlType, nullable = true, metadata)

      structFieldArray(indexCounter) = struckField
      indexCounter += 1
    }

    val result = new StructType(structFieldArray)
    result
  }

  /**
   * Here we are building the functionality to populate the resulting RDD[Row]
   * Here is where we will do the following:
   * - Filter push down
   * - Scan or GetList pruning
   * - Executing our scan(s) or/and GetList to generate result
   *
   * @param requiredColumns The columns that are being requested by the requesting query
   * @param filters         The filters that are being applied by the requesting query
   * @return                RDD will all the results from HBase needed for SparkSQL to
   *                        execute the query on
   */
  override def buildScan(requiredColumns: Array[String], filters: Array[Filter]): RDD[Row] = {

    //retain the information for unit testing checks
    var resultRDD: RDD[Row] = null

    if (resultRDD == null) {

      val strBuilder = new StringBuilder()
      var isFirst = true
      requiredColumns.foreach( c => {
        if (isFirst) isFirst = false
        else strBuilder.append(",")
        strBuilder.append(c)
      })

      val rdd = kuduContext.kuduRDD(tableName, strBuilder.toString()).map(r => {

        val rowResults = r._2
        Row.fromSeq(requiredColumns.map(c =>
          getKuduValue(c, rowResults)))
      })

      resultRDD=rdd
    }
    resultRDD
  }

  def getKuduValue(columnName:String, row:RowResult): Any = {

    val columnSchema = getKuduSchemaColumnMap.getOrElse(columnName, null)

    val columnType = row.getColumnType(columnName)

    if (columnType == Type.BINARY) row.getBinary(columnName)
    else if (columnType == Type.BOOL) row.getBoolean(columnName)
    else if (columnType == Type.DOUBLE) row.getDouble(columnName)
    else if (columnType == Type.FLOAT) row.getFloat(columnName)
    else if (columnType == Type.INT16) row.getShort(columnName)
    else if (columnType == Type.INT32) row.getInt(columnName)
    else if (columnType == Type.INT64) row.getLong(columnName)
    else if (columnType == Type.INT8) row.getByte(columnName)
    else if (columnType == Type.TIMESTAMP) row.getLong(columnName)
    else if (columnType == Type.STRING) row.getString(columnName)
  }
}
