package is.hail.io

import java.nio.ByteBuffer

import com.datastax.driver.core.querybuilder.QueryBuilder
import com.datastax.driver.core.{Cluster, DataType, Session, TableMetadata, Row => CassRow}
import is.hail.expr._
import is.hail.keytable.KeyTable
import is.hail.utils._
import org.apache.spark.sql.Row
import org.json4s._
import org.json4s.jackson.JsonMethods

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.util.Random

object CassandraImpex {
  def importSchema(table: TableMetadata): TStruct =
    TStruct(
      table.getColumns.asScala
        .map(col => (col.getName, importType(col.getType))): _*)

  def importRow(table: TableMetadata, row: CassRow): Any = {
    val a = new Array[Any](table.getColumns.size)

    val tableColIndex = table.getColumns.asScala
      .map(col => col.getName)
      .zipWithIndex
      .toMap

    val rowColumns = row.getColumnDefinitions

    val n = rowColumns.size
    var i = 0
    while (i < n) {
      val colName = rowColumns.getName(i)
      val j = tableColIndex(colName)
      a(j) = importAnnotation(row.getObject(i), rowColumns.getType(i))
      i += 1
    }

    Row.fromSeq(a)
  }

  def exportType(t: Type): DataType = exportType(t, 0)

  def exportType(t: Type, depth: Int): DataType = t match {
    case _: TBoolean => DataType.cboolean()
    case _: TInt32 => DataType.cint()
    case _: TInt64 => DataType.bigint()
    case _: TFloat32 => DataType.cfloat()
    case _: TFloat64 => DataType.cdouble()
    case _: TString => DataType.text()
    case _: TBinary => DataType.blob()
    case _: TCall => DataType.cint()
    case TArray(elementType, _) => DataType.list(exportType(elementType, depth + 1), depth == 1)
    case TSet(elementType, _) => DataType.set(exportType(elementType, depth + 1), depth == 1)
    case TDict(keyType, valueType, _) =>
      DataType.map(exportType(keyType, depth + 1),
        exportType(valueType, depth + 1), depth == 1)
    case _: TAltAllele => DataType.text()
    case TVariant(_, _) => DataType.text()
    case TLocus(_, _) => DataType.text()
    case TInterval(_, _) => DataType.text()
    case s: TStruct => DataType.text()
  }

  def importType(dt: DataType): Type = {
    (dt.getName: @unchecked) match {
      case DataType.Name.BOOLEAN => TBoolean()
      case DataType.Name.ASCII | DataType.Name.TEXT | DataType.Name.VARCHAR => TString()
      case DataType.Name.TINYINT => TInt32()
      case DataType.Name.SMALLINT => TInt32()
      case DataType.Name.INT => TInt32()
      case DataType.Name.BIGINT | DataType.Name.COUNTER => TInt64()
      case DataType.Name.FLOAT => TFloat32()
      case DataType.Name.DOUBLE => TFloat64()

      case DataType.Name.LIST =>
        val typeArgs = dt.getTypeArguments
        assert(typeArgs.size() == 1)
        TArray(importType(typeArgs.get(0)))

      case DataType.Name.SET =>
        val typeArgs = dt.getTypeArguments
        assert(typeArgs.size() == 1)
        TSet(importType(typeArgs.get(0)))

      case DataType.Name.MAP =>
        val typeArgs = dt.getTypeArguments
        assert(typeArgs.size() == 2)
        TDict(
          importType(typeArgs.get(0)),
          importType(typeArgs.get(1)))

      // FIXME nice message
    }
  }

  def exportAnnotation(a: Any, t: Type): Any = t match {
    case _: TBoolean => a
    case _: TInt32 => a
    case _: TInt64 => a
    case _: TFloat32 => a
    case _: TFloat64 => a
    case _: TString => a
    case _: TBinary => ByteBuffer.wrap(a.asInstanceOf[Array[Byte]])
    case _: TCall => a
    case TArray(elementType, _) =>
      if (a == null)
        null
      else
        a.asInstanceOf[Seq[_]].map(x => exportAnnotation(x, elementType)).asJava
    case TSet(elementType, _) =>
      if (a == null)
        null
      else
        a.asInstanceOf[Set[_]].map(x => exportAnnotation(x, elementType)).asJava
    case TDict(keyType, valueType, _) =>
      if (a == null)
        null
      else
        a.asInstanceOf[Map[_, _]].map { case (k, v) =>
          (exportAnnotation(k, keyType),
            exportAnnotation(v, valueType))
        }.asJava
    case _: TAltAllele | TVariant(_, _) | TLocus(_, _) | TInterval(_, _) =>
      JsonMethods.compact(t.toJSON(a))
    case s: TStruct => DataType.text()
      JsonMethods.compact(t.toJSON(a))
  }

  def importAnnotation(a: Any, dt: DataType): Any =
    if (a == null)
      null
    else {
      (dt.getName: @unchecked) match {
        case DataType.Name.BOOLEAN => a
        case DataType.Name.ASCII | DataType.Name.TEXT | DataType.Name.VARCHAR => a
        case DataType.Name.TINYINT => a.asInstanceOf[java.lang.Byte].toInt
        case DataType.Name.SMALLINT => a.asInstanceOf[java.lang.Short].toInt
        case DataType.Name.INT => a
        case DataType.Name.BIGINT | DataType.Name.COUNTER => a
        case DataType.Name.FLOAT => a
        case DataType.Name.DOUBLE => a

        case DataType.Name.LIST =>
          val typeArgs = dt.getTypeArguments
          assert(typeArgs.size() == 1)
          val elementDataType = typeArgs.get(0)

          a.asInstanceOf[java.util.List[_]].asScala
            .map(x => importAnnotation(x, elementDataType))
            .toArray[Any]: IndexedSeq[Any]

        case DataType.Name.SET =>
          val typeArgs = dt.getTypeArguments
          assert(typeArgs.size() == 1)
          val elementDataType = typeArgs.get(0)
          a.asInstanceOf[java.util.Set[_]].asScala
            .map(x => importAnnotation(x, elementDataType))
            .toSet

        case DataType.Name.MAP =>
          val typeArgs = dt.getTypeArguments
          assert(typeArgs.size() == 2)
          val keyDataType = typeArgs.get(0)
          val valueDataType = typeArgs.get(1)
          a.asInstanceOf[java.util.Map[_, _]].asScala
            .map { case (k, v) =>
              (importAnnotation(k, keyDataType),
                importAnnotation(v, valueDataType))
            }.toMap
      }
    }
}

object CassandraConnector {
  private var cluster: Cluster = _
  private var session: Session = _

  private var refcount: Int = 0

  def getSession(address: String): Session = {
    this.synchronized {
      if (cluster == null)
        cluster = Cluster.builder()
          .addContactPoint(address)
          .build()

      if (session == null)
        session = cluster.connect()

      refcount += 1
    }

    session
  }

  def disconnect() {
    this.synchronized {
      refcount -= 1
      if (refcount == 0) {
        session.close()
        cluster.close()

        session = null
        cluster = null
      }
    }
  }

  def pause(nano: Long) {
    if (nano > 0) {
      Thread.sleep((nano / 1000000).toInt, (nano % 1000000).toInt)
    }
  }

  def export(kt: KeyTable,
    address: String, keyspace: String, table: String,
    blockSize: Int = 100, rate: Int = 1000) {

    val sc = kt.hc.sc

    val qualifiedTable = keyspace + "." + table

    val fields = kt.signature.fields.map { f => (f.name, f.typ) }

    val session = CassandraConnector.getSession(address)

    var keyspaceMetadata = session.getCluster.getMetadata.getKeyspace(keyspace)
    var tableMetadata = keyspaceMetadata.getTable(table)

    /*
    // get keyspace (create it if it doesn't exist)
    if (keyspaceMetadata == null) {
      info(s"creating keyspace ${ keyspace }")
      try {
        session.execute(s"CREATE KEYSPACE ${ keyspace } " +
          "WITH replication = {'class': 'SimpleStrategy', 'replication_factor': '1'}  AND durable_writes = true")
      } catch {
        case e: Exception => fatal(s"exportvariantscass: unable to create keyspace ${ keyspace }: ${ e }")
      }
      keyspaceMetadata = session.getCluster.getMetadata.getKeyspace(keyspace)
    }

    // get table (drop and create it if necessary)
    if (drop) {
      info(s"dropping table ${ qualifiedTable }")
      try {
        session.execute(SchemaBuilder.dropTable(keyspace, table).ifExists());
      } catch {
        case e: Exception => warn(s"exportvariantscass: unable to drop table ${ qualifiedTable }: ${ e }")
      }
    }

    var tableMetadata = keyspaceMetadata.getTable(table)
    if (tableMetadata == null) {
      info(s"creating table ${ qualifiedTable }")
      try {
        session.execute(s"CREATE TABLE $qualifiedTable (${ escapeString("dataset_id") } text, chrom text, start int, ref text, alt text, " +
          s"PRIMARY KEY ((${ escapeString("dataset_id") }, chrom, start), ref, alt))") // WITH COMPACT STORAGE")
      } catch {
        case e: Exception => fatal(s"exportvariantscass: unable to create table ${ qualifiedTable }: ${ e }")
      }
      //info(s"table ${qualifiedTable} created")
      tableMetadata = keyspaceMetadata.getTable(table)
    } */

    val preexistingFields = tableMetadata.getColumns.asScala.map(_.getName).toSet
    val toAdd = fields
      .filter { case (name, t) => !preexistingFields(name) }

    if (toAdd.nonEmpty) {
      session.execute(s"ALTER TABLE $qualifiedTable ADD (${
        toAdd.map { case (name, t) => s""""$name" ${ CassandraImpex.exportType(t) }""" }.mkString(",")
      })")
    }

    CassandraConnector.disconnect()

    val localSignature = kt.signature
    val localBlockSize = blockSize
    val maxRetryInterval = 3 * 60 * 1000 // 3m

    val minInsertTimeNano = 1000000000L / rate

    kt.rdd.foreachPartition { it =>
      val session = CassandraConnector.getSession(address)
      val nb = new mutable.ArrayBuffer[String]
      val vb = new mutable.ArrayBuffer[AnyRef]

      var lastInsertNano = System.nanoTime()

      it
        .grouped(localBlockSize)
        .foreach { block =>

          var retryInterval = 3 * 1000 // 3s

          var toInsert = block.map { r =>
            nb.clear()
            vb.clear()

            (r.asInstanceOf[Row].toSeq, localSignature.fields).zipped
              .map { case (a, f) =>
                if (a != null) {
                  nb += "\"" + f.name + "\""
                  vb += CassandraImpex.exportAnnotation(a, f.typ).asInstanceOf[AnyRef]
                }
              }

            (nb.toArray, vb.toArray)
          }

          while (toInsert.nonEmpty) {
            val nano = System.nanoTime()
            pause(minInsertTimeNano - (nano - lastInsertNano))

            toInsert = toInsert.map { case nv@(names, values) =>
              val future = session.executeAsync(QueryBuilder
                .insertInto(keyspace, table)
                .values(names, values)
                // .enableTracing()
              )

              (nv, future)
            }.flatMap { case (nv, future) =>
              try {
                val rs = future.getUninterruptibly()
                // val micros = rs.getExecutionInfo.getQueryTrace.getDurationMicros
                // println(s"time: ${ micros }us")
                None
              } catch {
                case t: Throwable =>
                  warn(s"caught exception while adding inserting: ${
                    expandException(t, logMessage = true)
                  }\n\tretrying")

                  Some(nv)
              }
            }

            if (toInsert.nonEmpty) {
              Thread.sleep(Random.nextInt(retryInterval))
              retryInterval = (retryInterval * 2).max(maxRetryInterval)
            }
          }
        }

      CassandraConnector.disconnect()
    }
  }
}
