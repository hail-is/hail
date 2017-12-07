package is.hail.keytable

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr._
import is.hail.io.annotators.{BedAnnotator, IntervalList}
import is.hail.io.plink.{FamFileConfig, PlinkLoader}
import is.hail.io.{CassandraConnector, SolrConnector, exportTypes}
import is.hail.methods.{Aggregators, Filter}
import is.hail.rvd.RVD
import is.hail.utils._
import is.hail.variant.{GenomeReference, VSMLocalValue}
import org.apache.commons.lang3.StringUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.storage.StorageLevel
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization

import scala.collection.JavaConverters._
import scala.language.implicitConversions
import scala.reflect.ClassTag

sealed abstract class SortOrder

case object Ascending extends SortOrder

case object Descending extends SortOrder

object SortColumn {
  implicit def fromColumn(column: String): SortColumn = SortColumn(column, Ascending)
}

case class SortColumn(column: String, sortOrder: SortOrder)

case class KeyTableMetadata(
  version: Int,
  key: Array[String],
  schema: String,
  globalSchema: Option[String],
  globals: Option[JValue],
  n_partitions: Int)

case class KTLocalValue(globals: Row)

object KeyTable {
  final val fileVersion: Int = 0x101

  def range(hc: HailContext, n: Int, partitions: Option[Int] = None): KeyTable = {
    val range = Range(0, n).view.map(Row(_))
    val rdd = partitions match {
      case Some(parts) => hc.sc.parallelize(range, numSlices = parts)
      case None => hc.sc.parallelize(range)
    }
    KeyTable(hc, rdd, TStruct("index" -> TInt32()), Array("index"))
  }

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
    if (!hc.hadoopConf.exists(metadataFile))
      fatal(
        s"corrupt KeyTable: metadata file does not exist: $metadataFile")

    val metadata = hc.hadoopConf.readFile(metadataFile) { in =>
      // FIXME why doesn't this work?  Serialization.read[KeyTableMetadata](in)
      val json = parse(in)
      json.extract[KeyTableMetadata]
    }

    val schema = Parser.parseType(metadata.schema).asInstanceOf[TStruct]
    val globalSchema = metadata.globalSchema.map(str => Parser.parseType(str).asInstanceOf[TStruct]).getOrElse(TStruct.empty())
    val globals = metadata.globals.map(g => JSONAnnotationImpex.importAnnotation(g, globalSchema).asInstanceOf[Row])
      .getOrElse(Row.empty)
    KeyTable(hc,
      hc.readRows(path, schema, metadata.n_partitions)
        .map { rv =>
          new UnsafeRow(schema, rv.region.copy(), rv.offset): Row
        },
      schema,
      metadata.key,
      globalSchema,
      globals)
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

  def importIntervalList(hc: HailContext, filename: String,
    gr: GenomeReference = GenomeReference.defaultReference): KeyTable = {
    IntervalList.read(hc, filename, gr)
  }

  def importBED(hc: HailContext, filename: String,
    gr: GenomeReference = GenomeReference.defaultReference): KeyTable = {
    BedAnnotator.apply(hc, filename, gr)
  }

  def importFam(hc: HailContext, path: String, isQuantitative: Boolean = false,
    delimiter: String = "\\t",
    missingValue: String = "NA"): KeyTable = {

    val ffConfig = FamFileConfig(isQuantitative, delimiter, missingValue)

    val (data, typ) = PlinkLoader.parseFam(path, ffConfig, hc.hadoopConf)

    val rows = data.map { case (id, values) => Row.fromSeq(Array(id) ++ values.asInstanceOf[Row].toSeq) }.toArray
    val rdd = hc.sc.parallelize(rows)

    val newFields = List("ID" -> TString()) ++ typ.asInstanceOf[TStruct].fields.map(f => (f.name, f.typ))
    val struct = TStruct(newFields: _*)

    KeyTable(hc, rdd, struct, Array("ID"))
  }

  def apply(hc: HailContext, rdd: RDD[Row], signature: TStruct, key: Array[String] = Array.empty,
    globalSignature: TStruct = TStruct.empty(), globals: Row = Row.empty): KeyTable = {
    val rdd2 = rdd.mapPartitions { it =>
      val region = MemoryBuffer()
      val rvb = new RegionValueBuilder(region)
      val rv = RegionValue(region)
      it.map { row =>
        region.clear()
        rvb.start(signature)
        rvb.addAnnotation(signature, row)
        rv.setOffset(rvb.end())
        rv
      }
    }

    new KeyTable(hc, KeyTableLiteral(
      KeyTableType(signature, key, globalSignature),
      KeyTableValue(KeyTableType(signature, key, globalSignature), KTLocalValue(globals), RVD(signature, rdd2))
      ))
  }
}

class KeyTable(val hc: HailContext,
  val ir: KeyTableIR) {

  def this(hc: HailContext, rdd: RDD[RegionValue], signature: TStruct, key: Array[String] = Array.empty,
    globalSignature: TStruct = TStruct.empty(), globals: Row = Row.empty) = this(hc, KeyTableLiteral(
    KeyTableType(signature, key, globalSignature),
    KeyTableValue(KeyTableType(signature, key, globalSignature), KTLocalValue(globals), RVD(signature, rdd))
  ))

  lazy val value: KeyTableValue = {
    val opt = KeyTableIR.optimize(ir)
    opt.execute(hc)
  }

  lazy val KeyTableValue(ktType, KTLocalValue(globals), rvd) = value

  val KeyTableType(signature, key, globalSignature) = ir.typ

  lazy val rdd: RDD[Row] = value.rdd

  if (!(columns ++ globalSignature.fieldNames).areDistinct())
    fatal(s"Column names are not distinct: ${ (columns ++ globalSignature.fieldNames).duplicates().mkString(", ") }")
  if (!key.areDistinct())
    fatal(s"Key names are not distinct: ${ key.duplicates().mkString(", ") }")
  if (!key.forall(columns.contains(_)))
    fatal(s"Key names found that are not column names: ${ key.filterNot(columns.contains(_)).mkString(", ") }")

  private def rowEvalContext(): EvalContext = {
    val ec = EvalContext((fields.map { f => f.name -> f.typ } ++
      globalSignature.fields.map { f => f.name -> f.typ }): _*)
    var i = 0
    while (i < globalSignature.size) {
      ec.set(i + nColumns, globals.get(i))
      i += 1
    }
    ec
  }

  def fields: Array[Field] = signature.fields.toArray

  def keyFields: Array[Field] = key.map(signature.fieldIdx).map(i => fields(i))

  def columns: Array[String] = fields.map(_.name)

  def count(): Long = rvd.count()

  def nColumns: Int = fields.length

  def nKeys: Int = key.length

  def nPartitions: Int = rvd.partitions.length

  def keySignature: TStruct = {
    val (t, _) = signature.select(key)
    t
  }

  def valueSignature: TStruct = {
    val (t, _) = signature.filter(key.toSet, include = false)
    t
  }

  def typeCheck() {

    if (!globalSignature.typeCheck(globals)) {
      fatal(
        s"""found violation of global signature
           |  Schema: ${ globalSignature.toPrettyString(compact = true) }
           |  Annotation: $globals""".stripMargin)
    }

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
    val fieldIndices = fields.map(f => f.name -> f.index).toMap
    val keyIndices = key.map(fieldIndices)
    val keyIndexSet = keyIndices.toSet
    val valueIndices = fields.filter(f => !keyIndexSet.contains(f.index)).map(_.index)
    rdd.map { r => (Row.fromSeq(keyIndices.map(r.get)), Row.fromSeq(valueIndices.map(r.get))) }
  }

  def same(other: KeyTable): Boolean = {
    if (signature != other.signature) {
      info(
        s"""different signatures:
           | left: ${ signature.toPrettyString(compact = true) }
           | right: ${ other.signature.toPrettyString(compact = true) }
           |""".stripMargin)
      false
    } else if (key.toSeq != other.key.toSeq) {
      info(
        s"""different key names:
           | left: ${ key.mkString(", ") }
           | right: ${ other.key.mkString(", ") }
           |""".stripMargin)
      false
    } else if (globalSignature != other.globalSignature) {
      info(
        s"""different global signatures:
           | left: ${ globalSignature.toPrettyString(compact = true) }
           | right: ${ other.globalSignature.toPrettyString(compact = true) }
           |""".stripMargin)
      false
    } else if (globals != other.globals) {
      info(
        s"""different global annotations:
           | left: $globals
           | right: ${ other.globals }
           |""".stripMargin)
      false
    } else {
      keyedRDD().groupByKey().fullOuterJoin(other.keyedRDD().groupByKey()).forall { case (k, (v1, v2)) =>
        (v1, v2) match {
          case (None, None) => true
          case (Some(x), Some(y)) =>
            val r1 = x.toArray
            val r2 = y.toArray
            val res = if (r1.length != r2.length)
              false
            else r1.counter() == r2.counter()
            if (!res)
              info(s"SAME KEY, DIFFERENT VALUES: k=$k\n  left:\n    ${ r1.mkString("\n    ") }\n  right:\n    ${ r2.mkString("\n    ") }")
            res
          case _ =>
            info(s"KEY MISMATCH: k=$k\n  left=$v1\n  right=$v2")
            false
        }
      }
    }
  }

  def mapAnnotations[T](f: (Row) => T)(implicit tct: ClassTag[T]): RDD[T] = rdd.map(r => f(r))

  def query(expr: String): (Annotation, Type) = query(Array(expr)).head

  def query(exprs: java.util.ArrayList[String]): Array[(Annotation, Type)] = query(exprs.asScala.toArray)

  def query(exprs: Array[String]): Array[(Annotation, Type)] = {
    val aggregationST = (fields.map { f => f.name -> f.typ } ++
      globalSignature.fields.map { f => f.name -> f.typ })
      .zipWithIndex
      .map { case ((name, t), i) => name -> (i, t) }
      .toMap

    val ec = EvalContext((fields.map { f => f.name -> TAggregable(f.typ, aggregationST) } ++
      globalSignature.fields.map { f => f.name -> f.typ })
      .zipWithIndex
      .map { case ((name, t), i) => name -> (i, t) }
      .toMap)

    val g = globals.asInstanceOf[Row]
    var i = 0
    while (i < globalSignature.size) {
      ec.set(nColumns + i, g.get(i))
      i += 1
    }

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
    val ec = rowEvalContext()
    val (t, f) = Parser.parseExpr(code, ec)

    val f2: (Annotation) => Any = { a =>
      ec.setAllFromRow(a.asInstanceOf[Row])
      f()
    }

    (t, f2)
  }

  def annotateGlobal(a: Annotation, t: Type, name: String): KeyTable = {
    val (newT, i) = globalSignature.insert(t, name)
    copy2(globalSignature = newT.asInstanceOf[TStruct], globals = i(globals, a).asInstanceOf[Row])
  }

  def annotateGlobalExpr(expr: String): KeyTable = {
    val ec = EvalContext(globalSignature.fields.map(fd => (fd.name, fd.typ)): _*)

    ec.setAllFromRow(globals.asInstanceOf[Row])
    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, None)

    val inserterBuilder = new ArrayBuilder[Inserter]()

    val finalType = (paths, types).zipped.foldLeft(globalSignature) { case (v, (ids, signature)) =>
      val (s, i) = v.insert(signature, ids)
      inserterBuilder += i
      s.asInstanceOf[TStruct]
    }

    val inserters = inserterBuilder.result()

    val ga = inserters
      .zip(f())
      .foldLeft(globals) { case (a, (ins, res)) =>
        ins(a, res).asInstanceOf[Row]
      }

    copy2(globals = ga,
      globalSignature = finalType)
  }

  def selectGlobal(exprs: java.util.ArrayList[String]): KeyTable = {
    val ec = EvalContext(globalSignature.fields.map(fd => (fd.name, fd.typ)): _*)
    ec.setAllFromRow(globals.asInstanceOf[Row])

    val (maybePaths, types, f, isNamedExpr) = Parser.parseSelectExprs(exprs.asScala.toArray, ec)

    val paths = maybePaths.zip(isNamedExpr).map { case (p, isNamed) =>
      if (isNamed)
        List(p.mkString(".")) // FIXME: Not certain what behavior we want here
      else {
          List(p.last)
      }
    }

    val overlappingPaths = paths.counter().filter { case (n, i) => i != 1 }.keys

    if (overlappingPaths.nonEmpty)
      fatal(s"Found ${ overlappingPaths.size } ${ plural(overlappingPaths.size, "selected field name") } that are duplicated.\n" +
        "Overlapping fields:\n  " +
        s"@1", overlappingPaths.truncatable("\n  "))

    val inserterBuilder = new ArrayBuilder[Inserter]()

    val finalSignature = (paths, types).zipped.foldLeft(TStruct()) { case (vs, (p, sig)) =>
      val (s: TStruct, i) = vs.insert(sig, p)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()

    val newGlobal = f().zip(inserters)
      .foldLeft(Row()) { case (a1, (v, inserter)) =>
        inserter(a1, v).asInstanceOf[Row]
      }

    copy2(globalSignature = finalSignature, globals = newGlobal)
  }

  def annotate(cond: String): KeyTable = {
    val ec = rowEvalContext()

    val (paths, types, f) = Parser.parseAnnotationExprs(cond, ec, None)

    val inserterBuilder = new ArrayBuilder[Inserter]()

    val finalSignature = (paths, types).zipped.foldLeft(signature) { case (vs, (ids, sig)) =>
      val (s: TStruct, i) = vs.insert(sig, ids)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()

    val annotF: Row => Row = { r =>
      ec.setAllFromRow(r)

      f().zip(inserters)
        .foldLeft(r) { case (a1, (v, inserter)) =>
          inserter(a1, v).asInstanceOf[Row]
        }
    }

    copy(rdd = mapAnnotations(annotF), signature = finalSignature, key = key)
  }

  def filter(p: Annotation => Boolean): KeyTable = copy(rdd = rdd.filter(p))

  def filter(cond: String, keep: Boolean): KeyTable = {
    val ec = rowEvalContext()

    val f: () => java.lang.Boolean = Parser.parseTypedExpr[java.lang.Boolean](cond, ec)

    val p = (a: Annotation) => {
      ec.setAllFromRow(a.asInstanceOf[Row])
      Filter.boxedKeepThis(f(), keep)
    }

    filter(p)
  }

  def head(n: Long): KeyTable = {
    if (n < 0)
      fatal(s"n must be non-negative! Found `$n'.")
    copy(rdd = rdd.head(n))
  }

  def keyBy(key: String*): KeyTable = keyBy(key)

  def keyBy(key: java.util.ArrayList[String]): KeyTable = keyBy(key.asScala)

  def keyBy(key: Iterable[String]): KeyTable = {
    val colSet = columns.toSet
    val badKeys = key.filter(!colSet.contains(_))

    if (badKeys.nonEmpty)
      fatal(
        s"""Invalid ${ plural(badKeys.size, "key") }: [ ${ badKeys.map(x => s"'$x'").mkString(", ") } ]
           |  Available columns: [ ${ signature.fields.map(x => s"'${ x.name }'").mkString(", ") } ]""".stripMargin)

    copy(key = key.toArray)
  }

  def select(exprs: Array[String], qualifiedName: Boolean = false): KeyTable = {
    val ec = rowEvalContext()

    val (maybePaths, types, f, isNamedExpr) = Parser.parseSelectExprs(exprs, ec)

    val paths = maybePaths.zip(isNamedExpr).map { case (p, isNamed) =>
      if (isNamed)
        List(p.mkString(".")) // FIXME: Not certain what behavior we want here
      else {
        if (qualifiedName)
          List(p.mkString("."))
        else
          List(p.last)
      }
    }

    val overlappingPaths = paths.counter.filter { case (n, i) => i != 1 }.keys

    if (overlappingPaths.nonEmpty)
      fatal(s"Found ${ overlappingPaths.size } ${ plural(overlappingPaths.size, "selected field name") } that are duplicated.\n" +
        "Either rename manually or use the 'mangle' option to handle duplicates.\n Overlapping fields:\n  " +
        s"@1", overlappingPaths.truncatable("\n  "))

    val inserterBuilder = new ArrayBuilder[Inserter]()

    val finalSignature = (paths, types).zipped.foldLeft(TStruct()) { case (vs, (p, sig)) =>
      val (s: TStruct, i) = vs.insert(sig, p)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()

    val annotF: Row => Row = { r =>
      ec.setAllFromRow(r)

      f().zip(inserters)
        .foldLeft(Row()) { case (a1, (v, inserter)) =>
          inserter(a1, v).asInstanceOf[Row]
        }
    }

    val newKey = key.filter(paths.map(_.mkString(".")).toSet)

    copy(rdd = mapAnnotations(annotF), signature = finalSignature, key = newKey)
  }

  def select(selectedColumns: String*): KeyTable = select(selectedColumns.toArray)

  def select(selectedColumns: java.util.ArrayList[String], mangle: Boolean): KeyTable =
    select(selectedColumns.asScala.toArray, mangle)

  def drop(columnsToDrop: Array[String]): KeyTable = {
    val nonexistentColumns = columnsToDrop.diff(columns)
    if (nonexistentColumns.nonEmpty)
      fatal(s"Columns `${ nonexistentColumns.mkString(", ") }' do not exist in key table. Choose from `${ columns.mkString(", ") }'.")

    val selectedColumns = columns.diff(columnsToDrop)
    select(selectedColumns)
  }

  def drop(columnsToDrop: java.util.ArrayList[String]): KeyTable = drop(columnsToDrop.asScala.toArray)

  def rename(columnMap: Map[String, String]): KeyTable = {
    val newSignature = TStruct(signature.fields.map { fd => fd.copy(name = columnMap.getOrElse(fd.name, fd.name)) })
    val newColumns = newSignature.fields.map(_.name)
    val newKey = key.map(n => columnMap.getOrElse(n, n))
    val duplicateColumns = newColumns.foldLeft(Map[String, Int]() withDefaultValue 0) { (m, x) => m + (x -> (m(x) + 1)) }.filter {
      _._2 > 1
    }

    if (duplicateColumns.nonEmpty)
      fatal(s"Found duplicate column names after renaming columns: `${ duplicateColumns.keys.mkString(", ") }'")

    copy(rdd = rdd, signature = newSignature, key = newKey)
  }

  def rename(newColumns: Array[String]): KeyTable = {
    if (newColumns.length != nColumns)
      fatal(s"Found ${ newColumns.length } new column names but need $nColumns.")

    rename((columns, newColumns).zipped.toMap)
  }

  def rename(columnMap: java.util.HashMap[String, String]): KeyTable = rename(columnMap.asScala.toMap)

  def rename(newColumns: java.util.ArrayList[String]): KeyTable = rename(newColumns.asScala.toArray)

  def join(other: KeyTable, joinType: String): KeyTable = {
    if (key.length != other.key.length || !(keyFields.map(_.typ) sameElements other.keyFields.map(_.typ)))
      fatal(
        s"""Both key tables must have the same number of keys and the types of keys must be identical. Order matters.
           |  Left signature: ${ keySignature.toPrettyString(compact = true) }
           |  Right signature: ${ other.keySignature.toPrettyString(compact = true) }""".stripMargin)

    val joinedFields = keySignature.fields ++ valueSignature.fields ++ other.valueSignature.fields

    val preNames = joinedFields.map(_.name).toArray
    val (finalColumnNames, remapped) = mangle(preNames)
    if (remapped.nonEmpty) {
      warn(s"Remapped ${ remapped.length } ${ plural(remapped.length, "column") } from right-hand table:\n  @1",
        remapped.map { case (pre, post) => s""""$pre" => "$post"""" }.truncatable("\n  "))
    }

    val newSignature = TStruct(joinedFields
      .zipWithIndex
      .map { case (fd, i) => (finalColumnNames(i), fd.typ) }: _*)
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

    copy(rdd = joinedRDD, signature = newSignature, key = key)
  }

  def forall(code: String): Boolean = {
    val ec = rowEvalContext()

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
    val ec = rowEvalContext()
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

  def export(output: String, typesFile: String = null, header: Boolean = true, parallel: Boolean = false) {
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
          sb.append(TableAnnotationImpex.exportAnnotation(r.get(i), localTypes(i)))
        }(sb += '\t')

        sb.result()
      }
    }.writeTable(output, hc.tmpDir, Some(fields.map(_.name).mkString("\t")).filter(_ => header), parallelWrite = parallel)
  }

  def aggregate(keyCond: String, aggCond: String, nPartitions: Option[Int] = None): KeyTable = {

    val aggregationST = (fields.map { f => f.name -> f.typ } ++
      globalSignature.fields.map { f => f.name -> f.typ })
      .zipWithIndex
      .map { case ((name, t), i) => name -> (i, t) }
      .toMap

    val ec = EvalContext((fields.map { f => f.name -> TAggregable(f.typ, aggregationST) } ++
      globalSignature.fields.map { f => f.name -> f.typ })
      .zipWithIndex
      .map { case ((name, t), i) => name -> (i, t) }
      .toMap)

    val keyEC = EvalContext(aggregationST)

    val g = globals.asInstanceOf[Row]
    var i = 0
    while (i < globalSignature.size) {
      ec.set(nColumns + i, g.get(i))
      keyEC.set(nColumns + i, g.get(i))
      i += 1
    }

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
    }.aggregateByKey(zVals, nPartitions.getOrElse(this.nPartitions))(seqOp, combOp)
      .map {
        case (k, agg) =>
          resultOp(agg)
          Row.fromSeq(k.toSeq ++ aggF())
      }

    copy(rdd = newRDD, signature = keySignature.merge(aggSignature)._1, key = newKey)
  }

  def ungroup(column: String, mangle: Boolean = false): KeyTable = {
    val (finalSignature, ungrouper) = signature.ungroup(column, mangle)
    copy(rdd = rdd.map(ungrouper), signature = finalSignature)
  }

  def group(dest: String, columns: java.util.ArrayList[String]): KeyTable = group(dest, columns.asScala.toArray)

  def group(dest: String, columns: Array[String]): KeyTable = {
    val (newSignature, grouper) = signature.group(dest, columns)
    val newKey = key.filter(!columns.contains(_))
    copy(rdd = rdd.map(grouper), signature = newSignature, key = newKey)
  }


  def expandTypes(): KeyTable = {
    val localSignature = signature
    val expandedSignature = Annotation.expandType(localSignature).asInstanceOf[TStruct]

    copy(rdd = rdd.map { a => Annotation.expandAnnotation(a, localSignature).asInstanceOf[Row] },
      signature = expandedSignature,
      key = key)
  }

  def flatten(): KeyTable = {
    val localSignature = signature
    val keySignature = TStruct(keyFields)
    val flattenedSignature = Annotation.flattenType(localSignature).asInstanceOf[TStruct]
    val flattenedKey = Annotation.flattenType(keySignature).asInstanceOf[TStruct].fields.map(_.name).toArray

    copy(rdd = rdd.map { a => Annotation.flattenAnnotation(a, localSignature).asInstanceOf[Row] },
      signature = flattenedSignature,
      key = flattenedKey)
  }

  def toDF(sqlContext: SQLContext): DataFrame = {
    val localSignature = signature
    sqlContext.createDataFrame(
      rdd.map {
        a => SparkAnnotationImpex.exportAnnotation(a, localSignature).asInstanceOf[Row]
      },
      signature.schema.asInstanceOf[StructType])
  }

  def explode(columnToExplode: String): KeyTable = {

    val explodeField = signature.fieldOption(columnToExplode) match {
      case Some(x) => x
      case None =>
        fatal(
          s"""Input field name `${ columnToExplode }' not found in KeyTable.
             |KeyTable field names are `${ columns.mkString(", ") }'.""".stripMargin)
    }

    val index = explodeField.index

    val explodeType = explodeField.typ match {
      case t: TIterable => t.elementType
      case _ => fatal(s"Require Array or Set. Column `$columnToExplode' has type `${ explodeField.typ }'.")
    }

    val newSignature = signature.copy(fields = fields.updated(index, Field(columnToExplode, explodeType, index)))

    val empty = Iterable.empty[Row]
    val explodedRDD = rdd.flatMap { a =>
      val row = a.toSeq
      val it = row(index)
      if (it == null)
        empty
      else
        for (element <- row(index).asInstanceOf[Iterable[_]]) yield Row.fromSeq(row.updated(index, element))
    }

    copy(rdd = explodedRDD, signature = newSignature, key = key)
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

    val metadata = KeyTableMetadata(KeyTable.fileVersion,
      key,
      signature.toPrettyString(compact = true),
      Some(globalSignature.toPrettyString(compact = true)),
      Some(JSONAnnotationImpex.exportAnnotation(globals, globalSignature)),
      nPartitions)
    hc.hadoopConf.writeTextFile(path + "/metadata.json.gz")(out =>
      Serialization.write(metadata, out))

    rvd.rdd.writeRows(path, signature)
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

  def unpersist() {
    rdd.unpersist()
  }

  def orderBy(sortCols: SortColumn*): KeyTable =
    orderBy(sortCols.toArray)

  def orderBy(sortCols: Array[SortColumn]): KeyTable = {
    val sortColIndexOrd = sortCols.map { case SortColumn(n, so) =>
      val i = signature.fieldIdx(n)
      val f = signature.fields(i)

      val fo = f.typ.ordering(so == Ascending)

      (i, if (so == Ascending) fo else fo.reverse)
    }

    val ord: Ordering[Annotation] = new Ordering[Annotation] {
      def compare(a: Annotation, b: Annotation): Int = {
        var i = 0
        while (i < sortColIndexOrd.length) {
          val (fi, ford) = sortColIndexOrd(i)
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

  def repartition(n: Int, shuffle: Boolean = true): KeyTable = copy(rdd = rdd.coalesce(n, shuffle))

  def union(kts: java.util.ArrayList[KeyTable]): KeyTable = union(kts.asScala.toArray: _*)

  def union(kts: KeyTable*): KeyTable = {
    kts.foreach { kt =>
      if (signature != kt.signature)
        fatal("cannot union tables with different schemas")
      if (!key.sameElements(kt.key))
        fatal("cannot union tables with different key")
    }

    copy(rdd = hc.sc.union(rdd, kts.map(_.rdd): _*))
  }

  def take(n: Int): Array[Row] = rdd.take(n)

  def indexed(name: String = "index"): KeyTable = {
    if (columns.contains(name))
      fatal(s"name collision: cannot index table, because column '$name' already exists")

    val (newSignature, ins) = signature.insert(TInt64(), name)

    val newRDD = rdd.zipWithIndex().map { case (r, ind) => ins(r, ind).asInstanceOf[Row] }

    copy(signature = newSignature.asInstanceOf[TStruct], rdd = newRDD)
  }

  def maximalIndependentSet(iExpr: String, jExpr: String, tieBreakerExpr: Option[String] = None): Array[Any] = {
    val ec = rowEvalContext()

    val (iType, iThunk) = Parser.parseExpr(iExpr, ec)
    val (jType, jThunk) = Parser.parseExpr(jExpr, ec)

    if (iType != jType)
      fatal(s"node expressions must have the same type: type of `i' is $iType, but type of `j' is $jType")

    val tieBreakerEc = EvalContext("l" -> iType, "r" -> iType)

    val maybeTieBreaker = tieBreakerExpr.map { e =>
      val tieBreakerThunk = Parser.parseTypedExpr[Int](e, tieBreakerEc)

      (l: Any, r: Any) => {
        tieBreakerEc.setAll(l, r)
        tieBreakerThunk()
      }
    }.getOrElse(null)

    val edgeRdd = mapAnnotations { r =>
      ec.setAllFromRow(r)
      (iThunk(), jThunk())
    }

    if (edgeRdd.count() > 400000)
      warn(s"over 400,000 edges are in the graph; maximal_independent_set may run out of memory")

    Graph.maximalIndependentSet(edgeRdd.collect(), maybeTieBreaker)
  }

  def showString(n: Int = 10, truncate: Option[Int] = None, printTypes: Boolean = true): String = {
    /**
      * Parts of this method are lifted from:
      *   org.apache.spark.sql.Dataset.showString
      * Spark version 2.0.2
      */

    require(n >= 0, s"number of rows to show must be non-negative, found $n")
    truncate.foreach { tr => require(tr > 3, s"truncation length too small: $tr") }

    val takeResult = take(n + 1)
    val hasMoreData = takeResult.length > n
    val data = takeResult.take(n)

    def convertType(t: Type, name: String, ab: ArrayBuilder[(String, String, Boolean)]) {
      t match {
        case s: TStruct => s.fields.foreach { f =>
          convertType(f.typ, if (name == null) f.name else name + "." + f.name, ab)
        }
        case _ =>
          ab += (name, t.toPrettyString(compact = true), t.isInstanceOf[TNumeric])
      }
    }

    val headerBuilder = new ArrayBuilder[(String, String, Boolean)]()
    convertType(signature, null, headerBuilder)
    val (names, types, rightAlign) = headerBuilder.result().unzip3

    def convertValue(t: Type, v: Annotation, ab: ArrayBuilder[String]) {
      t match {
        case s: TStruct =>
          val r = v.asInstanceOf[Row]
          s.fields.foreach(f => convertValue(f.typ, if (r == null) null else r.get(f.index), ab))
        case _ =>
          ab += t.str(v)
      }
    }

    val valueBuilder = new ArrayBuilder[String]()
    val dataStrings = data.map { r =>
      valueBuilder.clear()
      convertValue(signature, r, valueBuilder)
      valueBuilder.result()
    }

    val allStrings = (Iterator(names, types) ++ dataStrings.iterator).map { arr =>
      arr.map { str =>
        truncate match {
          case Some(tr) if str.length > tr => str.substring(0, tr - 3) + "..."
          case _ => str
        }
      }
    }.toArray

    // Initialize the width of each column to a minimum value of '3'
    val nCols = names.length
    val colWidths = Array.fill(nCols)(3)

    // Compute the width of each column
    for (i <- allStrings.indices)
      for (j <- 0 until nCols)
        colWidths(j) = math.max(colWidths(j), allStrings(i)(j).length)

    // create separator
    val sep: String = colWidths.map("-" * _).addString(new StringBuilder(), "+-", "-+-", "-+\n").toString()

    val sb = new StringBuilder()
    sb.clear()
    sb.append(sep)

    // column names
    allStrings(0).zipWithIndex.map { case (cell, i) =>
      if (rightAlign(i))
        StringUtils.leftPad(cell, colWidths(i))
      else
        StringUtils.rightPad(cell, colWidths(i))
    }.addString(sb, "| ", " | ", " |\n")

    sb.append(sep)

    if (printTypes) {
      // column types
      allStrings(1).zipWithIndex.map { case (cell, i) =>
        if (rightAlign(i))
          StringUtils.leftPad(cell, colWidths(i))
        else
          StringUtils.rightPad(cell, colWidths(i))
      }.addString(sb, "| ", " | ", " |\n")

      sb.append(sep)
    }

    // data
    allStrings.drop(2).foreach {
      _.zipWithIndex.map { case (cell, i) =>
        if (rightAlign(i))
          StringUtils.leftPad(cell, colWidths(i))
        else
          StringUtils.rightPad(cell, colWidths(i))
      }.addString(sb, "| ", " | ", " |\n")
    }

    sb.append(sep)

    if (hasMoreData)
      sb.append(s"showing top $n ${ plural(n, "row") }\n")

    sb.result()
  }

  def copy(rdd: RDD[Row] = rdd,
    signature: TStruct = signature,
    key: Array[String] = key,
    globalSignature: TStruct = globalSignature,
    globals: Row = globals): KeyTable = {
    KeyTable(hc, rdd, signature, key, globalSignature, globals)
  }

  def copy2(rvd: RVD = rvd,
    signature: TStruct = signature,
    key: Array[String] = key,
    globalSignature: TStruct = globalSignature,
    globals: Row = globals): KeyTable = {
    new KeyTable(hc, KeyTableLiteral(
      KeyTableType(signature, key, globalSignature),
      KeyTableValue(KeyTableType(signature, key, globalSignature), KTLocalValue(globals), rvd)
    ))
  }
}
