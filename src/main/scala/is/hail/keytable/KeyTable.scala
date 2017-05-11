package is.hail.keytable

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.{TStruct, _}
import is.hail.io.annotators.{BedAnnotator, IntervalList}
import is.hail.io.plink.{FamFileConfig, PlinkLoader}
import is.hail.io.{CassandraConnector, SolrConnector, exportTypes}
import is.hail.methods.{Aggregators, Filter}
import is.hail.utils._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.storage.StorageLevel
import org.json4s.JObject
import org.json4s.JsonAST._
import org.json4s.jackson.{JsonMethods, Serialization}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.language.implicitConversions
import scala.reflect.ClassTag

sealed abstract class SortOrder

case object Ascending extends SortOrder

case object Descending extends SortOrder

object SortColumn {
  implicit def fromField(field: String): SortColumn = SortColumn(field, Ascending)
}

case class SortColumn(field: String, sortOrder: SortOrder)

object KeyTable {
  final val fileVersion: Int = 1

  def fromDF(hc: HailContext, df: DataFrame, key: java.util.ArrayList[String]): KeyTable = {
    fromDF(hc, df, key.asScala.toArray)
  }

  def fromDF(hc: HailContext, df: DataFrame, key: Array[String] = Array.empty[String]): KeyTable = {
    val signature = SparkAnnotationImpex.importType(df.schema).asInstanceOf[TStruct]
    KeyTable(hc, df.rdd.map { r =>
      SparkAnnotationImpex.importAnnotation(r, signature).asInstanceOf[Row]
    },
      signature, key)
  }

  def read(hc: HailContext, path: String): KeyTable = {
    if (!hc.hadoopConf.exists(path))
      fatal(s"$path does not exist")
    else if (!path.endsWith(".kt") && !path.endsWith(".kt/"))
      fatal(s"key table files must end in '.kt', but found '$path'")

    val metadataFile = path + "/metadata.json.gz"
    val pqtSuccess = path + "/rdd.parquet/_SUCCESS"

    if (!hc.hadoopConf.exists(pqtSuccess))
      fatal(
        s"corrupt KeyTable: parquet success file does not exist: $pqtSuccess")

    if (!hc.hadoopConf.exists(metadataFile))
      fatal(
        s"corrupt KeyTable: metadata file does not exist: $metadataFile")

    val (signature, key) = try {
      val json = hc.hadoopConf.readFile(metadataFile)(in =>
        JsonMethods.parse(in))

      val fields = json.asInstanceOf[JObject].obj.toMap

      (fields.get("version"): @unchecked) match {
        case Some(JInt(v)) =>
          if (v != KeyTable.fileVersion)
            fatal(
              s"""Invalid KeyTable: old version
                 |  got version $v, expected version ${ KeyTable.fileVersion }""".stripMargin)
      }

      val signature = (fields.get("schema"): @unchecked) match {
        case Some(JString(s)) =>
          Parser.parseType(s).asInstanceOf[TStruct]
      }

      val key = (fields.get("key_names"): @unchecked) match {
        case Some(JArray(a)) =>
          a.map { case JString(s) => s }.toArray[String]
      }

      (signature, key)
    } catch {
      case e: Throwable =>
        fatal(
          s"""
             |corrupt KeyTable: invalid metadata file.
             |  caught exception: ${ expandException(e, logMessage = true) }
          """.stripMargin)
    }

    val requiresConversion = SparkAnnotationImpex.requiresConversion(signature)
    val parquetFile = path + "/rdd.parquet"

    val rdd = hc.sqlContext.read.parquet(parquetFile)
      .rdd
      .map { r =>
        if (requiresConversion)
          SparkAnnotationImpex.importAnnotation(r, signature).asInstanceOf[Row]
        else
          r
      }

    KeyTable(hc, rdd, signature, key)
  }

  def parallelize(hc: HailContext, rows: java.util.ArrayList[Row], signature: TStruct,
    keyNames: java.util.ArrayList[String], nPartitions: Option[Int]): KeyTable = {
    val sc = hc.sc
    KeyTable(hc,
      nPartitions match {
        case Some(n) =>
          sc.parallelize(rows.asScala, n)
        case None =>
          sc.parallelize(rows.asScala)
      }, signature, keyNames.asScala.toArray)
  }

  def importIntervalList(hc: HailContext, filename: String): KeyTable = {
    IntervalList.read(hc, filename)
  }

  def importBED(hc: HailContext, filename: String): KeyTable = {
    BedAnnotator.apply(hc, filename)
  }

  def importFam(hc: HailContext, path: String, isQuantitative: Boolean = false,
    delimiter: String = "\\t",
    missingValue: String = "NA"): KeyTable = {

    val ffConfig = FamFileConfig(isQuantitative, delimiter, missingValue)

    val (data, typ) = PlinkLoader.parseFam(path, ffConfig, hc.hadoopConf)

    val rows = data.map { case (id, values) => Row.fromSeq(Array(id) ++ values.asInstanceOf[Row].toSeq)}.toArray
    val rdd = hc.sc.parallelize(rows)

    val newFields = List("ID" -> TString) ++ typ.asInstanceOf[TStruct].fields.map(f => (f.name, f.typ))
    val struct = TStruct(newFields: _*)

    KeyTable(hc, rdd, struct, Array("ID"))
  }
}

case class KeyTable(hc: HailContext, rdd: RDD[Row],
  signature: TStruct, key: Array[String] = Array.empty) {

  if (!fieldNames.areDistinct())
    fatal(s"Column names are not distinct: ${ fieldNames.duplicates().mkString(", ") }")
  if (!key.areDistinct())
    fatal(s"Key names are not distinct: ${ key.duplicates().mkString(", ") }")
  if (!key.forall(fieldNames.contains(_)))
    fatal(s"Key names found that are not column names: ${ key.filterNot(fieldNames.contains(_)).mkString(", ") }")

  def fields: Array[Field] = signature.fields.toArray

  def keyFields: Array[Field] = key.map(signature.fieldIdx).map(i => fields(i))

  def fieldNames: Array[String] = fields.map(_.name)

  def nRows: Long = rdd.count()

  def nFields: Int = fields.length

  def nKeys: Int = key.length

  def nPartitions: Int = rdd.partitions.length

  def keySignature: TStruct = {
    val keySet = key.toSet
    TStruct(signature.fields.filter(f => keySet.contains(f.name)).map(f => (f.name, f.typ)): _*)
  }

  def valueSignature: TStruct = {
    val keySet = key.toSet
    TStruct(signature.fields.filter(f => !keySet.contains(f.name)).map(f => (f.name, f.typ)): _*)
  }

  def typeCheck() {
    val localSignature = signature
    rdd.foreach { a =>
      if (!localSignature.typeCheck(a))
        fatal(
          s"""found violation in row annotation
             |  Schema: ${ localSignature.toPrettyString() }
             |
             |  Annotation: ${ Annotation.printAnnotation(a) }""".stripMargin
        )
    }
  }

  def keyedRDD(): RDD[(Row, Row)] = {
    if (nKeys == 0)
      fatal("cannot produce a keyed RDD from a key table with no key columns")

    val keySet = keyFields.map(_.name).toSet
    val keyIndices = fields.filter(f => keySet.contains(f.name)).map(_.index)
    val valueIndices = fields.filter(f => !keySet.contains(f.name)).map(_.index)
    rdd.map { r => (Row.fromSeq(keyIndices.map(r.get)), Row.fromSeq(valueIndices.map(r.get))) }
  }

  def same(other: KeyTable): Boolean = {
    if (signature != other.signature) {
      info(
        s"""different signatures:
           | left: ${ signature.toPrettyString() }
           | right: ${ other.signature.toPrettyString() }
           |""".stripMargin)
      false
    } else if (key.toSeq != other.key.toSeq) {
      info(
        s"""different key names:
           | left: ${ key.mkString(", ") }
           | right: ${ other.key.mkString(", ") }
           |""".stripMargin)
      false
    } else {
      val thisFieldNames = fieldNames
      val otherFieldNames = other.fieldNames

      keyedRDD().groupByKey().fullOuterJoin(other.keyedRDD().groupByKey()).forall { case (k, (v1, v2)) =>
        (v1, v2) match {
          case (None, None) => true
          case (Some(x), Some(y)) =>
            val r1 = x.map(r => thisFieldNames.zip(r.toSeq).toMap).toSet
            val r2 = y.map(r => otherFieldNames.zip(r.toSeq).toMap).toSet
            val res = r1 == r2
            if (!res)
              info(s"k=$k r1=${ r1.mkString(",") } r2=${ r2.mkString(",") }")
            res
          case _ =>
            info(s"k=$k v1=$v1 v2=$v2")
            false
        }
      }
    }
  }

  def mapAnnotations[T](f: (Row) => T)(implicit tct: ClassTag[T]): RDD[T] = rdd.map(r => f(r))

  def query(expr: String): (Annotation, Type) = query(Array(expr)).head

  def query(exprs: java.util.ArrayList[String]): Array[(Annotation, Type)] = query(exprs.asScala.toArray)

  def query(exprs: Array[String]): Array[(Annotation, Type)] = {
    val aggregationST = fields.zipWithIndex.map {
      case (fd, i) => (fd.name, (i, fd.typ))
    }.toMap

    val ec = EvalContext(fields.zipWithIndex.map {
      case (fd, i) => (fd.name, (i, TAggregable(fd.typ, aggregationST)))
    }.toMap)

    val ts = exprs.map(e => Parser.parseExpr(e, ec))

    val (zVals, seqOp, combOp, resultOp) = Aggregators.makeFunctions[Annotation](ec, {
      case (ec, a) =>
        ec.setAllFromRow(a.asInstanceOf[Row])
    })

    val r = rdd.aggregate(zVals.map(_.copy()))(seqOp, combOp)
    resultOp(r)

    ts.map { case (t, f) => (f(), t) }
  }

  def queryRow(code: String): (Type, Querier) = {
    val ec = EvalContext(fields.map(f => (f.name, f.typ)): _*)
    val (t, f) = Parser.parseExpr(code, ec)

    val f2: (Annotation) => Any = { a =>
      ec.setAllFromRow(a.asInstanceOf[Row])
      f()
    }

    (t, f2)
  }

  def annotate(cond: String): KeyTable = {
    val ec = EvalContext(fields.map(fd => (fd.name, fd.typ)): _*)

    val (paths, types, f) = Parser.parseAnnotationExprs(cond, ec, None)

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]

    val finalSignature = (paths, types).zipped.foldLeft(signature) { case (vs, (ids, sig)) =>
      val (s: TStruct, i) = vs.insert(sig, ids)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()

    val nFieldsLocal = nFields

    val annotF: Row => Row = { r =>
      ec.setAllFromRow(r)

      f().zip(inserters)
        .foldLeft(r) { case (a1, (v, inserter)) =>
          inserter(a1, v).asInstanceOf[Row]
        }
    }

    KeyTable(hc, mapAnnotations(annotF), finalSignature, key)
  }

  def filter(p: Annotation => Boolean): KeyTable = copy(rdd = rdd.filter(p))

  def filter(cond: String, keep: Boolean): KeyTable = {
    val ec = EvalContext(fields.map(f => (f.name, f.typ)): _*)

    val f: () => java.lang.Boolean = Parser.parseTypedExpr[java.lang.Boolean](cond, ec)

    val p = (a: Annotation) => {
      ec.setAllFromRow(a.asInstanceOf[Row])
      Filter.boxedKeepThis(f(), keep)
    }

    filter(p)
  }

  def keyBy(key: String*): KeyTable = keyBy(key)

  def keyBy(key: java.util.ArrayList[String]): KeyTable = keyBy(key.asScala)

  def keyBy(key: Iterable[String]): KeyTable = {
    val colSet = fieldNames.toSet
    val badKeys = key.filter(!colSet.contains(_))

    if (badKeys.nonEmpty)
      fatal(
        s"""Invalid ${ plural(badKeys.size, "key") }: [ ${ badKeys.map(x => s"'$x'").mkString(", ") } ]
           |  Available columns: [ ${ signature.fields.map(x => s"'${ x.name }'").mkString(", ") } ]""".stripMargin)

    copy(key = key.toArray)
  }

  def select(fieldsSelect: Array[String], newKeys: Array[String]): KeyTable = {
    val keyColumnsNotInSelectedFields = newKeys.diff(fieldsSelect)
    if (keyColumnsNotInSelectedFields.nonEmpty)
      fatal(s"Key columns `${ keyColumnsNotInSelectedFields.mkString(", ") }' must be present in selected columns.")

    val fieldsNotExist = fieldsSelect.diff(fieldNames)
    if (fieldsNotExist.nonEmpty)
      fatal(s"Selected columns `${ fieldsNotExist.mkString(", ") }' do not exist in key table. Choose from `${ fieldNames.mkString(", ") }'.")

    val fieldTransform = fieldsSelect.map(cn => fieldNames.indexOf(cn))

    val newSignature = TStruct(fieldTransform.zipWithIndex.map { case (oldIndex, newIndex) => signature.fields(oldIndex).copy(index = newIndex) })

    val selectF: Row => Row = { r =>
      Row.fromSeq(fieldTransform.map(r.get))
    }

    KeyTable(hc, mapAnnotations(selectF), newSignature, newKeys)
  }

  def select(fieldsSelect: java.util.ArrayList[String], newKeys: java.util.ArrayList[String]): KeyTable =
    select(fieldsSelect.asScala.toArray, newKeys.asScala.toArray)

  def rename(fieldNameMap: Map[String, String]): KeyTable = {
    val newSignature = TStruct(signature.fields.map { fd => fd.copy(name = fieldNameMap.getOrElse(fd.name, fd.name)) })
    val newFieldNames = newSignature.fields.map(_.name)
    val newKey = key.map(n => fieldNameMap.getOrElse(n, n))
    val duplicateFieldNames = newFieldNames.foldLeft(Map[String, Int]() withDefaultValue 0) { (m, x) => m + (x -> (m(x) + 1)) }.filter {
      _._2 > 1
    }

    if (duplicateFieldNames.nonEmpty)
      fatal(s"Found duplicate column names after renaming fields: `${ duplicateFieldNames.keys.mkString(", ") }'")

    KeyTable(hc, rdd, newSignature, newKey)
  }

  def rename(newFieldNames: Array[String]): KeyTable = {
    if (newFieldNames.length != nFields)
      fatal(s"Found ${ newFieldNames.length } new column names but need $nFields.")

    rename((fieldNames, newFieldNames).zipped.toMap)
  }

  def rename(fieldNameMap: java.util.HashMap[String, String]): KeyTable = rename(fieldNameMap.asScala.toMap)

  def rename(newFieldNames: java.util.ArrayList[String]): KeyTable = rename(newFieldNames.asScala.toArray)

  def join(other: KeyTable, joinType: String): KeyTable = {
    if (key.length != other.key.length || !(keyFields.map(_.typ) sameElements other.keyFields.map(_.typ)))
      fatal(
        s"""Both key tables must have the same number of keys and the types of keys must be identical. Order matters.
           |  Left signature: ${ TStruct(keyFields).toPrettyString(compact = true) }
           |  Right signature: ${ TStruct(other.keyFields).toPrettyString(compact = true) }""".stripMargin)

    val overlappingFields = fieldNames.toSet.intersect(other.fieldNames.toSet) -- key -- other.key
    if (overlappingFields.nonEmpty)
      fatal(
        s"""Columns that are not keys cannot be present in both key tables.
           |  Overlapping fields: ${ overlappingFields.mkString(", ") }""".stripMargin)


    val newSignature = TStruct((keySignature.fields ++ valueSignature.fields ++ other.valueSignature.fields)
      .map(fd => (fd.name, fd.typ)): _*)
    val localNKeys = nKeys
    val size1 = valueSignature.size
    val size2 = other.valueSignature.size
    val totalSize = newSignature.size

    assert(totalSize == localNKeys + size1 + size2)

    val merger = (k: Row, r1: Row, r2: Row) => {
      val result = Array.fill[Any](totalSize)(null)

      var i = 0
      while (i < localNKeys) {
        result(i) = k.get(i)
        i += 1
      }

      if (r1 != null) {
        i = 0
        while (i < size1) {
          result(localNKeys + i) = r1.get(i)
          i += 1
        }
      }

      if (r2 != null) {
        i = 0
        while (i < size2) {
          result(localNKeys + size1 + i) = r2.get(i)
          i += 1
        }
      }
      Row.fromSeq(result)
    }

    val rddLeft = keyedRDD()
    val rddRight = other.keyedRDD()

    val joinedRDD = joinType match {
      case "left" => rddLeft.leftOuterJoin(rddRight).map { case (k, (l, r)) => merger(k, l, r.orNull) }
      case "right" => rddLeft.rightOuterJoin(rddRight).map { case (k, (l, r)) => merger(k, l.orNull, r) }
      case "inner" => rddLeft.join(rddRight).map { case (k, (l, r)) => merger(k, l, r) }
      case "outer" => rddLeft.fullOuterJoin(rddRight).map { case (k, (l, r)) => merger(k, l.orNull, r.orNull) }
      case _ => fatal("Invalid join type specified. Choose one of `left', `right', `inner', `outer'")
    }

    KeyTable(hc, joinedRDD, newSignature, key)
  }

  def forall(code: String): Boolean = {
    val ec = EvalContext(fields.map(f => (f.name, f.typ)): _*)
    val f: () => java.lang.Boolean = Parser.parseTypedExpr[java.lang.Boolean](code, ec)(boxedboolHr)

    rdd.forall { a =>
      ec.setAllFromRow(a)
      val b = f()
      if (b == null)
        false
      else
        b
    }
  }

  def exists(code: String): Boolean = {
    val ec = EvalContext(fields.map(f => (f.name, f.typ)): _*)
    val f: () => java.lang.Boolean = Parser.parseTypedExpr[java.lang.Boolean](code, ec)(boxedboolHr)

    rdd.exists { r =>
      ec.setAllFromRow(r)
      val b = f()
      if (b == null)
        false
      else
        b
    }
  }

  def export(output: String, typesFile: String = null, header: Boolean = true) {
    val hConf = hc.hadoopConf
    hConf.delete(output, recursive = true)

    Option(typesFile).foreach { file =>
      exportTypes(file, hConf, fields.map(f => (f.name, f.typ)))
    }

    val localTypes = fields.map(_.typ)

    rdd.mapPartitions { it =>
      val sb = new StringBuilder()

      it.map { r =>
        sb.clear()

        localTypes.indices.foreachBetween { i =>
          sb.append(localTypes(i).str(r.get(i)))
        }(sb += '\t')

        sb.result()
      }
    }.writeTable(output, hc.tmpDir, Some(fields.map(_.name).mkString("\t")).filter(_ => header))
  }

  def aggregate(keyCond: String, aggCond: String): KeyTable = {

    val aggregationST = fields.zipWithIndex.map {
      case (fd, i) => (fd.name, (i, fd.typ))
    }.toMap

    val keyEC = EvalContext(aggregationST)
    val ec = EvalContext(fields.zipWithIndex.map {
      case (fd, i) => (fd.name, (i, TAggregable(fd.typ, aggregationST)))
    }.toMap)

    val (keyPaths, keyTypes, keyF) = Parser.parseAnnotationExprs(keyCond, keyEC, None)

    val (aggPaths, aggTypes, aggF) = Parser.parseAnnotationExprs(aggCond, ec, None)

    val newKey = keyPaths.map(_.head)
    val aggNames = aggPaths.map(_.head)

    val keySignature = TStruct((newKey, keyTypes).zipped.toSeq: _*)
    val aggSignature = TStruct((aggNames, aggTypes).zipped.toSeq: _*)

    val (zVals, seqOp, combOp, resultOp) = Aggregators.makeFunctions[Row](ec, {
      case (ec, r) =>
        ec.setAllFromRow(r)
    })

    val newRDD = rdd.mapPartitions {
      it =>
        it.map {
          r =>
            keyEC.setAllFromRow(r)
            val key = Row.fromSeq(keyF())
            (key, r)
        }
    }.aggregateByKey(zVals)(seqOp, combOp)
      .map {
        case (k, agg) =>
          resultOp(agg)
          Row.fromSeq(k.toSeq ++ aggF())
      }

    KeyTable(hc, newRDD, keySignature.merge(aggSignature)._1, newKey)
  }

  def expandTypes(): KeyTable = {
    val localSignature = signature
    val expandedSignature = Annotation.expandType(localSignature).asInstanceOf[TStruct]

    KeyTable(hc, rdd.map { a => Annotation.expandAnnotation(a, localSignature).asInstanceOf[Row] },
      expandedSignature,
      key)
  }

  def flatten(): KeyTable = {
    val localSignature = signature
    val keySignature = TStruct(keyFields)
    val flattenedSignature = Annotation.flattenType(localSignature).asInstanceOf[TStruct]
    val flattenedKey = Annotation.flattenType(keySignature).asInstanceOf[TStruct].fields.map(_.name).toArray

    KeyTable(hc, rdd.map { a => Annotation.flattenAnnotation(a, localSignature).asInstanceOf[Row] },
      flattenedSignature,
      flattenedKey)
  }

  def toDF(sqlContext: SQLContext): DataFrame = {
    val localSignature = signature
    sqlContext.createDataFrame(
      rdd.map {
        a => SparkAnnotationImpex.exportAnnotation(a, localSignature).asInstanceOf[Row]
      },
      signature.schema.asInstanceOf[StructType])
  }

  def explode(columnName: String): KeyTable = {

    val explodeField = signature.fieldOption(columnName) match {
      case Some(x) => x
      case None =>
        fatal(
          s"""Input field name `${ columnName }' not found in KeyTable.
             |KeyTable field names are `${ fieldNames.mkString(", ") }'.""".stripMargin)
    }

    val index = explodeField.index

    val explodeType = explodeField.typ match {
      case t: TIterable => t.elementType
      case _ => fatal(s"Require Array or Set. Column `$columnName' has type `${ explodeField.typ }'.")
    }

    val newSignature = signature.copy(fields = fields.updated(index, Field(columnName, explodeType, index)))

    val empty = Iterable.empty[Row]
    val explodedRDD = rdd.flatMap { a =>
      val row = a.toSeq
      val it = row(index)
      if (it == null)
        empty
      else
        for (element <- row(index).asInstanceOf[Iterable[_]]) yield Row.fromSeq(row.updated(index, element))
    }

    KeyTable(hc, explodedRDD, newSignature, key)
  }

  def explode(columnNames: Array[String]): KeyTable = {
    columnNames.foldLeft(this)((kt, name) => kt.explode(name))
  }

  def explode(columnNames: java.util.ArrayList[String]): KeyTable = explode(columnNames.asScala.toArray)

  def collect(): IndexedSeq[Annotation] = rdd.collect()

  def write(path: String, overwrite: Boolean = false) {
    if (!path.endsWith(".kt") && !path.endsWith(".kt/"))
      fatal(s"write path must end in '.kt', but found '$path'")

    if (overwrite)
      hc.hadoopConf.delete(path, recursive = true)
    else if (hc.hadoopConf.exists(path))
      fatal(s"$path already exists")

    hc.hadoopConf.mkDir(path)

    val sb = new StringBuilder
    sb.clear()
    signature.pretty(sb, printAttrs = true, compact = true)
    val schemaString = sb.result()

    val json = JObject(
      ("version", JInt(KeyTable.fileVersion)),
      ("key_names", JArray(key.map(k => JString(k)).toList)),
      ("schema", JString(schemaString)))

    hc.hadoopConf.writeTextFile(path + "/metadata.json.gz")(Serialization.writePretty(json, _))

    val localSignature = signature
    val requiresConversion = SparkAnnotationImpex.requiresConversion(signature)

    val rowRDD = rdd.map { a =>
      (if (requiresConversion)
        SparkAnnotationImpex.exportAnnotation(a, localSignature)
      else
        a).asInstanceOf[Row]
    }

    hc.sqlContext.createDataFrame(rowRDD, signature.schema.asInstanceOf[StructType])
      .write.parquet(path + "/rdd.parquet")
  }

  def cache(): KeyTable = persist("MEMORY_ONLY")

  def persist(storageLevel: String): KeyTable = {
    val level = try {
      StorageLevel.fromString(storageLevel)
    } catch {
      case e: IllegalArgumentException =>
        fatal(s"unknown StorageLevel `$storageLevel'")
    }

    rdd.persist(level)
    this
  }

  def orderBy(fields: SortColumn*): KeyTable =
    orderBy(fields.toArray)

  def orderBy(fields: Array[SortColumn]): KeyTable = {
    val fieldOrds = fields.map { case SortColumn(n, so) =>
      val i = signature.fieldIdx(n)
      val f = signature.fields(i)

      val fo = f.typ.ordering(so == Ascending)

      (i, if (so == Ascending) fo else fo.reverse)
    }

    val ord: Ordering[Annotation] = new Ordering[Annotation] {
      def compare(a: Annotation, b: Annotation): Int = {
        var i = 0
        while (i < fieldOrds.length) {
          val (fi, ford) = fieldOrds(i)
          val c = ford.compare(
            a.asInstanceOf[Row].get(fi),
            b.asInstanceOf[Row].get(fi))
          if (c != 0) return c
          i += 1
        }

        0
      }
    }

    val act = implicitly[ClassTag[Annotation]]
    copy(rdd = rdd.sortBy(identity[Annotation], ascending = true)(ord, act))
  }

  def exportSolr(zkHost: String, collection: String, blockSize: Int = 100): Unit = {
    SolrConnector.export(this, zkHost, collection, blockSize)
  }

  def exportCassandra(address: String, keyspace: String, table: String,
    blockSize: Int = 100, rate: Int = 1000): Unit = {
    CassandraConnector.export(this, address, keyspace, table, blockSize, rate)
  }

  def repartition(n: Int): KeyTable = copy(rdd = rdd.repartition(n))

  def union(other: KeyTable): KeyTable = {
    if (signature != other.signature)
      fatal("cannot union tables with different schemas")
    if (!key.sameElements(other.key))
      fatal("cannot union tables with different key")

    copy(rdd = rdd.union(other.rdd))
  }
}
