package is.hail.table

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr._
import is.hail.expr.ir
import is.hail.expr.ir.{IR, Pretty, TableAggregateByKey, TableExplode, TableFilter, TableIR, TableJoin, TableKeyBy, TableLiteral, TableMapGlobals, TableMapRows, TableOrderBy, TableParallelize, TableRange, TableRead, TableUnion, TableUnkey, TableValue}
import is.hail.expr.types._
import is.hail.io.plink.{FamFileConfig, LoadPlink}
import is.hail.methods.Aggregators
import is.hail.rvd._
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import is.hail.variant._
import org.apache.commons.lang3.StringUtils
import org.apache.spark.Partitioner
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.storage.StorageLevel
import org.json4s._
import org.json4s.jackson.JsonMethods

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.language.implicitConversions
import scala.reflect.ClassTag

sealed abstract class SortOrder

case object Ascending extends SortOrder

case object Descending extends SortOrder

case class SortField(field: String, sortOrder: SortOrder)

case class TableSpec(
  file_version: Int,
  hail_version: String,
  references_rel_path: String,
  table_type: TableType,
  components: Map[String, ComponentSpec]) extends RelationalSpec {
  def rowsComponent: RVDComponentSpec = getComponent[RVDComponentSpec]("rows")
}

object Table {
  def range(hc: HailContext, n: Int, nPartitions: Option[Int] = None): Table =
    new Table(hc, TableRange(n, nPartitions.getOrElse(hc.sc.defaultParallelism)))

  def fromDF(hc: HailContext, df: DataFrame, key: java.util.ArrayList[String]): Table = {
    fromDF(hc, df, if (key == null) None else Some(key.asScala.toArray.toFastIndexedSeq))
  }

  def fromDF(hc: HailContext, df: DataFrame, key: Option[IndexedSeq[String]] = None): Table = {
    val signature = SparkAnnotationImpex.importType(df.schema).asInstanceOf[TStruct]
    Table(hc, df.rdd, signature, key)
  }

  def read(hc: HailContext, path: String): Table =
    new Table(hc, TableIR.read(hc, path, dropRows = false, None))

  def parallelize(hc: HailContext, rowsJSON: String, signature: TStruct,
    keyNames: Option[java.util.ArrayList[String]], nPartitions: Option[Int]): Table = {
    val parsedRows = JSONAnnotationImpex.importAnnotation(JsonMethods.parse(rowsJSON), TArray(signature))
      .asInstanceOf[IndexedSeq[Row]]
    parallelize(hc, parsedRows, signature, keyNames.map(_.asScala.toArray.toFastIndexedSeq), nPartitions)
  }

  def parallelize(hc: HailContext, rows: IndexedSeq[Row], signature: TStruct,
    key: Option[IndexedSeq[String]], nPartitions: Option[Int]): Table = {
    val typ = TableType(signature, key.map(_.toArray.toFastIndexedSeq), TStruct())
    new Table(hc, TableParallelize(typ, rows, nPartitions))
  }

  def importFam(hc: HailContext, path: String, isQuantPheno: Boolean = false,
    delimiter: String = "\\t",
    missingValue: String = "NA"): Table = {

    val ffConfig = FamFileConfig(isQuantPheno, delimiter, missingValue)

    val (data, typ) = LoadPlink.parseFam(path, ffConfig, hc.hadoopConf)

    val rdd = hc.sc.parallelize(data)

    Table(hc, rdd, typ, Some(IndexedSeq("id")))
  }

  def apply(
    hc: HailContext,
    rdd: RDD[Row],
    signature: TStruct
  ): Table = apply(hc, rdd, signature, None, sort = true)

  def apply(
    hc: HailContext,
    rdd: RDD[Row],
    signature: TStruct,
    sort: Boolean
  ): Table = apply(hc, rdd, signature, None, sort)

  def apply(
    hc: HailContext,
    rdd: RDD[Row],
    signature: TStruct,
    key: Option[IndexedSeq[String]]
  ): Table = apply(hc, rdd, signature, key, TStruct.empty(), Annotation.empty, sort = true)

  def apply(
    hc: HailContext,
    rdd: RDD[Row],
    signature: TStruct,
    key: Option[IndexedSeq[String]],
    sort: Boolean
  ): Table = apply(hc, rdd, signature, key, TStruct.empty(), Annotation.empty, sort)

  def apply(
    hc: HailContext,
    rdd: RDD[Row],
    signature: TStruct,
    key: Option[IndexedSeq[String]],
    globalSignature: TStruct,
    globals: Annotation
  ): Table = apply(
    hc,
    ContextRDD.weaken[RVDContext](rdd),
    signature,
    key,
    globalSignature,
    globals,
    sort = true)

  def apply(
    hc: HailContext,
    rdd: RDD[Row],
    signature: TStruct,
    key: Option[IndexedSeq[String]],
    globalSignature: TStruct,
    globals: Annotation,
    sort: Boolean
  ): Table = apply(
    hc,
    ContextRDD.weaken[RVDContext](rdd),
    signature,
    key,
    globalSignature,
    globals,
    sort)

  def apply(
    hc: HailContext,
    crdd: ContextRDD[RVDContext, Row],
    signature: TStruct,
    key: Option[IndexedSeq[String]],
    sort: Boolean
  ): Table = apply(hc, crdd, signature, key, TStruct.empty(), Annotation.empty, sort)

  def apply(
    hc: HailContext,
    crdd: ContextRDD[RVDContext, Row],
    signature: TStruct,
    key: Option[IndexedSeq[String]],
    globalSignature: TStruct,
    globals: Annotation,
    sort: Boolean
  ): Table = {
    val crdd2 = crdd.cmapPartitions((ctx, it) => it.toRegionValueIterator(ctx.region, signature))
    new Table(hc, TableLiteral(
      TableValue(
        TableType(signature, None, globalSignature),
        BroadcastRow(globals.asInstanceOf[Row], globalSignature, hc.sc),
        new UnpartitionedRVD(signature, crdd2)))
    ).keyBy(key.map(_.toArray), sort)
  }

  def sameWithinTolerance(t: Type, l: Array[Row], r: Array[Row], tolerance: Double, absolute: Boolean): Boolean = {
    val used = new Array[Boolean](r.length)
    var i = 0
    while (i < l.length) {
      val li = l(i)
      var matched = false
      var j = 0
      while (!matched && j < l.length && !used(j)) {
        matched = t.valuesSimilar(li, r(j), tolerance, absolute)
        if (matched)
          used(j) = true
        j += 1
      }
      if (!matched)
        return false
      i += 1
    }
    return true
  }
}

class Table(val hc: HailContext, val tir: TableIR) {
  def this(
    hc: HailContext,
    crdd: ContextRDD[RVDContext, RegionValue],
    signature: TStruct,
    key: Option[IndexedSeq[String]] = None,
    globalSignature: TStruct = TStruct.empty(),
    globals: Row = Row.empty
  ) = this(hc,
    TableLiteral(
      TableValue(
        TableType(signature, key, globalSignature),
        BroadcastRow(globals, globalSignature, hc.sc),
        new UnpartitionedRVD(signature, crdd))))

  def typ: TableType = tir.typ
  
  lazy val value: TableValue = {
    log.info("in Table.value: pre-opt:\n" + ir.Pretty(tir))
    val opt = ir.Optimize(tir)
    log.info("in Table.value: post-opt:\n" + ir.Pretty(opt))

    opt.execute(hc)
  }

  lazy val TableValue(ktType, globals, rvd) = value

  val TableType(signature, key, globalSignature) = tir.typ

  val keyOrEmpty: IndexedSeq[String] = tir.typ.keyOrEmpty
  val keyOrNull: IndexedSeq[String] = tir.typ.keyOrNull

  lazy val rdd: RDD[Row] = value.rdd

  if (!(fieldNames ++ globalSignature.fieldNames).areDistinct())
    fatal(s"Column names are not distinct: ${ (fieldNames ++ globalSignature.fieldNames).duplicates().mkString(", ") }")
  if (key.exists(key => !key.areDistinct()))
    fatal(s"Key names are not distinct: ${ key.get.duplicates().mkString(", ") }")
  if (key.exists(key => !key.forall(fieldNames.contains(_))))
    fatal(s"Key names found that are not column names: ${ key.get.filterNot(fieldNames.contains(_)).mkString(", ") }")

  def rowEvalContext(): EvalContext = {
    val ec = EvalContext(
      "global" -> globalSignature,
      "row" -> signature)
    ec
  }

  def aggEvalContext(): EvalContext = {
    val ec = EvalContext("global" -> globalSignature,
      "AGG" -> aggType())
    ec
  }

  def aggType(): TAggregable = {
    val aggSymbolTable = Map(
      "global" -> (0, globalSignature),
      "row" -> (1, signature)
    )
    TAggregable(signature, aggSymbolTable)
  }

  def fields: Array[Field] = signature.fields.toArray

  val keyFieldIdx: Option[Array[Int]] =
    key.map(_.toArray.map(signature.fieldIdx))

  def keyFields: Option[Array[Field]] =
    key.map(_.toArray.map(signature.fieldIdx).map(i => fields(i)))

  val valueFieldIdx: Array[Int] =
    signature.fields.filter(f =>
      !keyOrEmpty.contains(f.name)
    ).map(_.index).toArray

  def fieldNames: Array[String] = fields.map(_.name)

  def partitionCounts(): IndexedSeq[Long] = {
    tir.partitionCounts match {
      case Some(counts) => counts
      case None => rvd.countPerPartition()
    }
  }

  def count(): Long = ir.Interpret[Long](ir.TableCount(tir))

  def forceCount(): Long = rvd.count()

  def nColumns: Int = fields.length

  def nKeys: Option[Int] = key.map(_.length)

  def nPartitions: Int = rvd.getNumPartitions

  def keySignature: Option[TStruct] = key.map { key =>
    val (t, _) = signature.select(key.toArray)
    t
  }

  def valueSignature: TStruct = {
    val (t, _) = signature.filterSet(keyOrEmpty.toSet, include = false)
    t
  }

  def typeCheck() {

    if (!globalSignature.typeCheck(globals.value)) {
      fatal(
        s"""found violation of global signature
           |  Schema: ${ globalSignature.toString }
           |  Annotation: ${ globals.value }""".stripMargin)
    }

    val localSignature = signature
    rdd.foreach { a =>
      if (!localSignature.typeCheck(a))
        fatal(
          s"""found violation in row annotation
             |  Schema: ${ localSignature.toString }
             |
             |  Annotation: ${ Annotation.printAnnotation(a) }""".stripMargin
        )
    }
  }

  def keyedRDD(): RDD[(Row, Row)] = {
    require(key.isDefined)
    val fieldIndices = fields.map(f => f.name -> f.index).toMap
    val keyIndices = key.get.map(fieldIndices)
    val keyIndexSet = keyIndices.toSet
    val valueIndices = fields.filter(f => !keyIndexSet.contains(f.index)).map(_.index)
    rdd.map { r => (Row.fromSeq(keyIndices.map(r.get)), Row.fromSeq(valueIndices.map(r.get))) }
  }

  def same(other: Table, tolerance: Double = defaultTolerance, absolute: Boolean = false): Boolean = {
    val localValueSignature = valueSignature

    val globalSignatureOpt = globalSignature.deepOptional()
    if (signature.deepOptional() != other.signature.deepOptional()) {
      info(
        s"""different signatures:
           | left: ${ signature.toString }
           | right: ${ other.signature.toString }
           |""".stripMargin)
      false
    } else if (key.isDefined != other.key.isDefined ||
               (key.isDefined && key.get != other.key.get)) {
      info(
        s"""different keys:
            | left: ${ key.map(_.mkString(", ")).getOrElse("None") }
            | right: ${ other.key.map(_.mkString(", ")).getOrElse("None")}
            |""".stripMargin)
      false
    } else if (globalSignatureOpt != other.globalSignature.deepOptional()) {
      info(
        s"""different global signatures:
           | left: ${ globalSignature.toString }
           | right: ${ other.globalSignature.toString }
           |""".stripMargin)
      false
    } else if (!globalSignatureOpt.valuesSimilar(globals.value, other.globals.value)) {
      info(
        s"""different global annotations:
           | left: ${ globals.value }
           | right: ${ other.globals.value }
           |""".stripMargin)
      false
    } else if (key.isDefined) {
      keyedRDD().groupByKey().fullOuterJoin(other.keyedRDD().groupByKey()).forall { case (k, (v1, v2)) =>
        (v1, v2) match {
          case (Some(x), Some(y)) =>
            val r1 = x.toArray
            val r2 = y.toArray
            val res = if (r1.length != r2.length)
              false
            else r1.counter() == r2.counter() ||
              Table.sameWithinTolerance(localValueSignature, r1, r2, tolerance, absolute)
            if (!res)
              info(s"SAME KEY, DIFFERENT VALUES: k=$k\n  left:\n    ${ r1.mkString("\n    ") }\n  right:\n    ${ r2.mkString("\n    ") }")
            res
          case _ =>
            info(s"KEY MISMATCH: k=$k\n  left=$v1\n  right=$v2")
            false
        }
      }
    } else {
      assert(key.isEmpty)
      if (signature.fields.isEmpty) {
        val leftCount = count()
        val rightCount = other.count()
        if (leftCount != rightCount)
          info(s"EMPTY SCHEMAS, BUT DIFFERENT LENGTHS: left=$leftCount\n  right=$rightCount")
        leftCount == rightCount
      } else {
        keyBy(signature.fieldNames).keyedRDD().groupByKey().fullOuterJoin(
          other.keyBy(other.signature.fieldNames).keyedRDD().groupByKey()
        ).forall { case (k, (v1, v2)) =>
          (v1, v2) match {
            case (Some(x), Some(y)) => x.size == y.size
            case (Some(x), None) =>
              info(s"ROW IN LEFT, NOT RIGHT: ${ x.mkString("\n    ") }\n")
              false
            case (None, Some(y)) =>
              info(s"ROW IN RIGHT, NOT LEFT: ${ y.mkString("\n    ") }\n")
              false
            case (None, None) =>
              assert(false)
              false
          }
        }
      }
    }
  }

  def aggregateJSON(expr: String): String = {
    val (value, t) = aggregate(expr)
    makeJSON(t, value)
  }

  def aggregate(expr: String): (Any, Type) = {
    val ec = aggEvalContext()

    val queryAST = Parser.parseToAST(expr, ec)
    (queryAST.toIROpt(Some("AGG" -> "row")): @unchecked) match {
      case Some(ir) =>
        aggregate(ir)
    }
  }

  def aggregate(query: IR): (Any, Type) = {
    val t = ir.TableAggregate(tir, query)
    (ir.Interpret(t, ir.Env.empty, FastIndexedSeq(), None), t.typ)
  }

  def annotateGlobal(a: Annotation, t: Type, name: String): Table = {
    val at = TStruct(name -> t)
    val value = BroadcastRow(Row(a), at, hc.sc)
    new Table(hc, TableMapGlobals(tir,
      ir.InsertFields(ir.Ref("global", tir.typ.globalType), FastSeq(name -> ir.GetField(ir.Ref(s"value", at), name))), value))
  }

  def annotateGlobalJSON(data: String, t: TStruct): Table = {
    val ann = JSONAnnotationImpex.importAnnotation(JsonMethods.parse(data), t)
    val value = BroadcastRow(ann.asInstanceOf[Row], t, hc.sc)
    new Table(hc, TableMapGlobals(tir,
      ir.InsertFields(ir.Ref("global", tir.typ.globalType),
        t.fieldNames.map(name => name -> ir.GetField(ir.Ref("value", t), name))),
      value))
  }

  def selectGlobal(expr: String): Table = {
    val ec = EvalContext("global" -> globalSignature)

    val ast = Parser.parseToAST(expr, ec)
    assert(ast.`type`.isInstanceOf[TStruct])

    (ast.toIROpt(): @unchecked) match {
      case Some(ir) =>
        new Table(hc, TableMapGlobals(tir, ir, BroadcastRow(Row(), TStruct(), hc.sc)))
    }
  }

  def filter(cond: String, keep: Boolean): Table = {
    val ec = rowEvalContext()
    var filterAST = Parser.parseToAST(cond, ec)
    val pred = filterAST.toIROpt()
    (pred: @unchecked) match {
      case Some(irPred) =>
        new Table(hc,
          TableFilter(tir, ir.filterPredicateWithKeep(irPred, keep))
        )
    }
  }

  def head(n: Long): Table = {
    if (n < 0)
      fatal(s"n must be non-negative! Found `$n'.")
    copy2(rvd = rvd.head(n))
  }

  def keyBy(key: String*): Table = keyBy(key.toArray, null)

  def keyBy(keys: java.util.ArrayList[String]): Table =
    keyBy(Option(keys).map(_.asScala.toArray), true)

  def keyBy(
    keys: java.util.ArrayList[String],
    partitionKeys: java.util.ArrayList[String]
  ): Table = keyBy(keys.asScala.toArray, partitionKeys.asScala.toArray)

  def keyBy(keys: Array[String]): Table = keyBy(keys, sort = true)

  def keyBy(keys: Array[String], sort: Boolean): Table = keyBy(keys, null, sort)

  def keyBy(maybeKeys: Option[Array[String]]): Table = keyBy(maybeKeys, true)

  def keyBy(maybeKeys: Option[Array[String]], sort: Boolean): Table =
    maybeKeys match {
      case Some(keys) => keyBy(keys, sort)
      case None => unkey()
    }

  def keyBy(keys: Array[String], partitionKeys: Array[String], sort: Boolean = true): Table =
    new Table(hc, TableKeyBy(tir, keys, Option(partitionKeys).map(_.length), sort))

  def unkey(): Table =
    new Table(hc, TableUnkey(tir))

  def select(expr: String, newKey: java.util.ArrayList[String], preservedKeyFields: java.lang.Integer): Table =
    select(expr, Option(newKey).map(_.asScala.toFastIndexedSeq), Option(preservedKeyFields).map(_.toInt))

  def select(expr: String, newKey: Option[IndexedSeq[String]], preservedKeyFields: Option[Int]): Table = {
    val ec = rowEvalContext()
    val ast = Parser.parseToAST(expr, ec)
    assert(ast.`type`.isInstanceOf[TStruct])

    (ast.toIROpt(): @unchecked) match {
      case Some(ir) =>
        new Table(hc, TableMapRows(tir, ir, newKey, preservedKeyFields))
    }
  }

  def join(other: Table, joinType: String): Table =
    new Table(hc, TableJoin(this.tir, other.tir, joinType))

  def export(path: String, typesFile: String = null, header: Boolean = true, exportType: Int = ExportType.CONCATENATED) {
    ir.Interpret(ir.TableExport(tir, path, typesFile, header, exportType))
  }

  def distinctByKey(): Table = {
    new Table(hc, ir.TableDistinct(tir))
  }

  def groupByKey(name: String): Table = {
    require(key.isDefined)
    val sorted = keyBy(key.get.toArray, sort = true)
    sorted.copy2(
      rvd = sorted.rvd.asInstanceOf[OrderedRVD].groupByKey(name),
      signature = keySignature.get ++ TStruct(name -> TArray(valueSignature)))
  }

  def jToMatrixTable(rowKeys: java.util.ArrayList[String],
    colKeys: java.util.ArrayList[String],
    rowFields: java.util.ArrayList[String],
    colFields: java.util.ArrayList[String],
    partitionKeys: java.util.ArrayList[String],
    nPartitions: java.lang.Integer): MatrixTable = {

    toMatrixTable(rowKeys.asScala.toArray, colKeys.asScala.toArray,
      rowFields.asScala.toArray, colFields.asScala.toArray,
      partitionKeys.asScala.toArray,
      Option(nPartitions).map(_.asInstanceOf[Int])
    )
  }

  def toMatrixTable(
    rowKeys: Array[String],
    colKeys: Array[String],
    rowFields: Array[String],
    colFields: Array[String],
    partitionKeys: Array[String],
    nPartitions: Option[Int] = None
  ): MatrixTable = {

    // no fields used twice
    val fieldsUsed = mutable.Set.empty[String]
    (rowKeys ++ colKeys ++ rowFields ++ colFields).foreach { f =>
      assert(!fieldsUsed.contains(f))
      fieldsUsed += f
    }

    val entryFields = fieldNames.filter(f => !fieldsUsed.contains(f))

    // need keys for rows and cols
    assert(rowKeys.nonEmpty)
    assert(colKeys.nonEmpty)

    // check partition key is appropriate and not empty
    assert(rowKeys.startsWith(partitionKeys))
    assert(partitionKeys.nonEmpty)

    val fullRowType = signature

    val colKeyIndices = colKeys.map(signature.fieldIdx(_))
    val colValueIndices = colFields.map(signature.fieldIdx(_))

    val localColData = rvd.mapPartitions { it =>
      val ur = new UnsafeRow(fullRowType)
      it.map { rv =>
        val colKey = SafeRow.selectFields(fullRowType, rv)(colKeyIndices)
        val colValues = SafeRow.selectFields(fullRowType, rv)(colValueIndices)
        colKey -> colValues
      }
    }.reduceByKey({ case (l, _) => l }) // poor man's distinctByKey
      .collect()

    val nCols = localColData.length
    info(s"found $nCols columns")

    val colIndexBc = hc.sc.broadcast(localColData.zipWithIndex
      .map { case ((k, _), i) => (k, i) }
      .toMap)

    val rowType = TStruct((rowKeys ++ rowFields).map(f => f -> signature.fieldByName(f).typ): _*)
    val colType = TStruct((colKeys ++ colFields).map(f => f -> signature.fieldByName(f).typ): _*)
    val entryType = TStruct(entryFields.map(f => f -> signature.fieldByName(f).typ): _*)

    val colDataConcat = localColData.map { case (keys, values) => Row.fromSeq(keys.toSeq ++ values.toSeq): Annotation }

    // allFieldIndices has all row + entry fields
    val allFieldIndices = rowKeys.map(signature.fieldIdx(_)) ++ rowFields.map(signature.fieldIdx(_)) ++ entryFields.map(signature.fieldIdx(_))

    // FIXME replace with field namespaces
    val INDEX_UID = "*** COL IDX ***"

    // row and entry fields, plus an integer index
    val rowEntryStruct = rowType ++ entryType ++ TStruct(INDEX_UID -> TInt32Optional)

    val rowEntryRVD = rvd.mapPartitions(rowEntryStruct) { it =>
      val ur = new UnsafeRow(fullRowType)
      val rvb = new RegionValueBuilder()
      val rv2 = RegionValue()

      it.map { rv =>
        rvb.set(rv.region)

        rvb.start(rowEntryStruct)
        rvb.startStruct()

        // add all non-col fields
        var i = 0
        while (i < allFieldIndices.length) {
          rvb.addField(fullRowType, rv, allFieldIndices(i))
          i += 1
        }

        // look up col key, replace with int index
        ur.set(rv)
        val colKey = Row.fromSeq(colKeyIndices.map(ur.get))
        val idx = colIndexBc.value(colKey)
        rvb.addInt(idx)

        rvb.endStruct()
        rv2.set(rv.region, rvb.end())
        rv2
      }
    }

    val ordType = new OrderedRVDType(partitionKeys, rowKeys ++ Array(INDEX_UID), rowEntryStruct)
    val ordered = OrderedRVD.coerce(ordType, rowEntryRVD)

    val matrixType: MatrixType = MatrixType.fromParts(
      globalSignature,
      colKeys,
      colType,
      partitionKeys,
      rowKeys,
      rowType,
      entryType)

    val orderedEntryIndices = entryFields.map(rowEntryStruct.fieldIdx)
    val orderedRKIndices = rowKeys.map(rowEntryStruct.fieldIdx)
    val orderedRowIndices = (rowKeys ++ rowFields).map(rowEntryStruct.fieldIdx)

    val idxIndex = rowEntryStruct.fieldIdx(INDEX_UID)
    assert(idxIndex == rowEntryStruct.size - 1)

    val newRVType = matrixType.rvRowType
    val orderedRKStruct = matrixType.rowKeyStruct

    val newRVD = ordered.boundary.mapPartitionsPreservesPartitioning(matrixType.orvdType, { (ctx, it) =>
      val region = ctx.region
      val rvb = ctx.rvb
      val outRV = RegionValue(region)

      OrderedRVIterator(
        new OrderedRVDType(partitionKeys, rowKeys, rowEntryStruct),
        it,
        ctx
      ).staircase.map { rowIt =>
        rvb.start(newRVType)
        rvb.startStruct()
        var i = 0
        while (i < orderedRowIndices.length) {
          rvb.addField(rowEntryStruct, rowIt.value, orderedRowIndices(i))
          i += 1
        }
        rvb.startArray(nCols)
        i = 0
        for (rv <- rowIt) {
          val nextInt = rv.region.loadInt(rowEntryStruct.fieldOffset(rv.offset, idxIndex))
          while (i < nextInt) {
            rvb.setMissing()
            i += 1
          }
          rvb.startStruct()
          var j = 0
          while (j < orderedEntryIndices.length) {
            rvb.addField(rowEntryStruct, rv, orderedEntryIndices(j))
            j += 1
          }
          rvb.endStruct()
          i += 1
        }
        while (i < nCols) {
          rvb.setMissing()
          i += 1
        }
        rvb.endArray()
        rvb.endStruct()
        outRV.setOffset(rvb.end())
        outRV
      }
    })
    new MatrixTable(hc,
      matrixType,
      globals,
      BroadcastIndexedSeq(colDataConcat, TArray(matrixType.colType), hc.sc),
      newRVD)
  }

  def aggregateByKey(expr: String, oldAggExpr: String, nPartitions: Option[Int] = None): Table = {
    val ec = aggEvalContext()
    val ast = Parser.parseToAST(expr, ec)

    (ast.toIROpt(Some("AGG" -> "row")): @unchecked) match {
      case Some(x) =>
        new Table(hc, TableAggregateByKey(tir, x))
    }
  }

  def expandTypes(): Table = {
    def deepExpand(t: Type): Type = {
      t match {
        case t: TContainer =>
          // set, dict => array
          TArray(deepExpand(t.elementType), t.required)
        case t: ComplexType =>
          deepExpand(t.representation).setRequired(t.required)
        case t: TBaseStruct =>
          // tuple => struct
          TStruct(t.required, t.fields.map { f => (f.name, deepExpand(f.typ)) }: _*)
        case _ => t
      }
    }

    val newRowType = deepExpand(signature).asInstanceOf[TStruct]
    copy2(
      rvd = new UnpartitionedRVD(newRowType, rvd.crdd),
      signature = newRowType)
  }

  def flatten(): Table = {
    def deepFlatten(t: TStruct, x: ir.IR): IndexedSeq[IndexedSeq[(String, ir.IR)]] = {
      t.fields.map { f =>
        f.typ match {
          case t: TStruct =>
            deepFlatten(t, ir.GetField(x, f.name))
              .flatten
              .map { case (n2, x2) =>
                f.name + "." + n2 -> x2
              }
          case _ =>
            IndexedSeq(f.name -> ir.GetField(x, f.name))
        }
      }
    }

    val newFields = deepFlatten(signature, ir.Ref("row", tir.typ.rowType))
    val newKey: Option[IndexedSeq[String]] = keyFieldIdx.map(_.flatMap { i =>
      newFields(i).map { case (n, _) => n }
    })
    val preservedKeyFields = keyFieldIdx.map(_.takeWhile(i => newFields(i).length == 1).length)

    new Table(hc, TableMapRows(tir, ir.MakeStruct(newFields.flatten), newKey, preservedKeyFields))
  }

  // expandTypes must be called before toDF
  def toDF(sqlContext: SQLContext): DataFrame = {
    val localSignature = signature
    sqlContext.createDataFrame(
      rvd.map { rv => SafeRow(localSignature, rv) },
      signature.schema.asInstanceOf[StructType])
  }

  def explode(column: String): Table = new Table(hc, TableExplode(tir, column))

  def explode(columnNames: Array[String]): Table = {
    columnNames.foldLeft(this)((kt, name) => kt.explode(name))
  }

  def explode(columnNames: java.util.ArrayList[String]): Table = explode(columnNames.asScala.toArray)

  def collect(): Array[Row] = rdd.collect()

  def collectJSON(): String = {
    val r = JSONAnnotationImpex.exportAnnotation(collect().toFastIndexedSeq, TArray(signature))
    JsonMethods.compact(r)
  }


  def write(path: String, overwrite: Boolean = false, codecSpecJSONStr: String = null) {
    ir.Interpret(ir.TableWrite(tir, path, overwrite, codecSpecJSONStr))
  }

  def cache(): Table = persist("MEMORY_ONLY")

  def persist(storageLevel: String): Table = {
    val level = try {
      StorageLevel.fromString(storageLevel)
    } catch {
      case e: IllegalArgumentException =>
        fatal(s"unknown StorageLevel `$storageLevel'")
    }

    copy2(rvd = rvd.persist(level))
  }

  def unpersist(): Table = copy2(rvd = rvd.unpersist())

  def orderBy(sortFields: Array[SortField]): Table = {
    new Table(hc, TableOrderBy(ir.TableUnkey(tir), sortFields))
  }

  def repartition(n: Int, shuffle: Boolean = true): Table = copy2(rvd = rvd.coalesce(n, shuffle))

  def union(kts: java.util.ArrayList[Table]): Table = union(kts.asScala.toArray: _*)

  def union(kts: Table*): Table = new Table(hc, TableUnion((tir +: kts.map(_.tir)).toFastIndexedSeq))

  def take(n: Int): Array[Row] = rvd.take(n, RVD.wireCodec)

  def takeJSON(n: Int): String = {
    val r = JSONAnnotationImpex.exportAnnotation(take(n).toFastIndexedSeq, TArray(signature))
    JsonMethods.compact(r)
  }

  def sample(p: Double, seed: Int = 1): Table = {
    require(p > 0 && p < 1, s"the 'p' parameter must fall between 0 and 1, found $p")
    copy2(rvd = rvd.sample(withReplacement = false, p, seed))
  }

  def index(name: String): Table = {
    if (fieldNames.contains(name))
      fatal(s"name collision: cannot index table, because column '$name' already exists")
    val newRvd = rvd.zipWithIndex(name)
    copy2(signature = newRvd.rowType, rvd = newRvd)
  }

  def show(n: Int = 10, truncate: Option[Int] = None, printTypes: Boolean = true, maxWidth: Int = 100): Unit = {
    println(showString(n, truncate, printTypes, maxWidth))
  }

  def showString(n: Int = 10, truncate: Option[Int] = None, printTypes: Boolean = true, maxWidth: Int = 100): String = {
    /**
      * Parts of this method are lifted from:
      *   org.apache.spark.sql.Dataset.showString
      * Spark version 2.0.2
      */

    truncate.foreach { tr => require(tr > 3, s"truncation length too small: $tr") }
    require(maxWidth >= 10, s"max width too small: $maxWidth")

    val (data, hasMoreData) = if (n < 0)
      collect() -> false
    else {
      val takeResult = take(n + 1)
      val hasMoreData = takeResult.length > n
      takeResult.take(n) -> hasMoreData
    }

    def convertType(t: Type, name: String, ab: ArrayBuilder[(String, String, Boolean)]) {
      t match {
        case s: TStruct => s.fields.foreach { f =>
          convertType(f.typ, if (name == null) f.name else name + "." + f.name, ab)
        }
        case _ =>
          ab += (name, t.toString, t.isInstanceOf[TNumeric])
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

    val fixedWidth = 4 // "| " + " |"
    val delimWidth = 3 // " | "

    val tr = truncate.getOrElse(maxWidth - 4)

    val allStrings = (Iterator(names, types) ++ dataStrings.iterator).map { arr =>
      arr.map { str => if (str.length > tr) str.substring(0, tr - 3) + "..." else str }
    }.toArray

    val nCols = names.length
    val colWidths = Array.fill(nCols)(0)

    // Compute the width of each column
    for (i <- allStrings.indices)
      for (j <- 0 until nCols)
        colWidths(j) = math.max(colWidths(j), allStrings(i)(j).length)

    val normedStrings = allStrings.map { line =>
      line.zipWithIndex.map { case (cell, i) =>
        if (rightAlign(i))
          StringUtils.leftPad(cell, colWidths(i))
        else
          StringUtils.rightPad(cell, colWidths(i))
      }
    }

    val sb = new StringBuilder()
    sb.clear()

    // writes cols [startIndex, endIndex)
    def writeCols(startIndex: Int, endIndex: Int) {

      val toWrite = (startIndex until endIndex).toArray

      val sep = toWrite.map(i => "-" * colWidths(i)).addString(new StringBuilder, "+-", "-+-", "-+\n").result()
      // add separator line
      sb.append(sep)

      // add column names
      toWrite.map(normedStrings(0)(_)).addString(sb, "| ", " | ", " |\n")

      // add separator line
      sb.append(sep)

      if (printTypes) {
        // add types
        toWrite.map(normedStrings(1)(_)).addString(sb, "| ", " | ", " |\n")

        // add separator line
        sb.append(sep)
      }

      // data
      normedStrings.drop(2).foreach {
        toWrite.map(_).addString(sb, "| ", " | ", " |\n")
      }

      // add separator line
      sb.append(sep)
    }

    if (nCols == 0)
      writeCols(0, 0)

    var colIdx = 0
    var first = true

    while (colIdx < nCols) {
      val startIdx = colIdx
      var colWidth = fixedWidth

      // consume at least one column, and take until the next column would put the width over maxWidth
      do {
        colWidth += 3 + colWidths(colIdx)
        colIdx += 1
      } while (colIdx < nCols && colWidth + delimWidth + colWidths(colIdx) <= maxWidth)

      if (!first) {
        sb.append('\n')
      }

      writeCols(startIdx, colIdx)

      first = false
    }

    if (hasMoreData)
      sb.append(s"showing top $n ${ plural(n, "row") }\n")

    sb.result()
  }

  def copy(rdd: RDD[Row] = rdd,
    signature: TStruct = signature,
    key: Option[IndexedSeq[String]] = key,
    globalSignature: TStruct = globalSignature,
    newGlobals: Annotation = globals.value): Table = {
    Table(hc, rdd, signature, key, globalSignature, newGlobals, sort = false)
  }

  def copy2(rvd: RVD = rvd,
    signature: TStruct = signature,
    key: Option[IndexedSeq[String]] = key,
    globalSignature: TStruct = globalSignature,
    globals: BroadcastRow = globals): Table = {
    new Table(hc, TableLiteral(
      TableValue(TableType(signature, key, globalSignature), globals, rvd)
    ))
  }
  def toOrderedRVD(hintPartitioner: Option[OrderedRVDPartitioner], partitionKeys: Int): OrderedRVD = {
    require(key.isDefined)
    val orderedKTType = new OrderedRVDType(key.get.take(partitionKeys).toArray, key.get.toArray, signature)
    assert(hintPartitioner.forall(p => p.pkType.types.sameElements(orderedKTType.pkType.types)))
    OrderedRVD.coerce(orderedKTType, rvd, None, hintPartitioner)
  }

  def intervalJoin(other: Table, fieldName: String): Table = {
    assert(other.keySignature.exists(s => s.size == 1 && s.types(0).isInstanceOf[TInterval]))
    val intervalType = other.keySignature.get.types(0).asInstanceOf[TInterval]
    assert(keySignature.exists(s => s.size == 1 && s.types(0) == intervalType.pointType))

    val leftORVD = rvd match {
      case ordered: OrderedRVD => ordered
      case unordered =>
        OrderedRVD.coerce(
          new OrderedRVDType(key.get.toArray, key.get.toArray, signature),
          unordered)
    }

    val typOrdering = intervalType.pointType.ordering

    val typToInsert: Type = other.valueSignature

    val (newRVType, ins) = signature.unsafeStructInsert(typToInsert, List(fieldName))

    val partBc = hc.sc.broadcast(leftORVD.partitioner)
    val rightSignature = other.signature
    val rightKeyFieldIdx = other.keyFieldIdx.get(0)
    val rightValueFieldIdx = other.valueFieldIdx
    val partitionKeyedIntervals = other.rvd.boundary.crdd
      .flatMap { rv =>
        val r = SafeRow(rightSignature, rv)
        val interval = r.getAs[Interval](rightKeyFieldIdx)
        if (interval != null) {
          val rangeTree = partBc.value.rangeTree
          val pkOrd = partBc.value.pkType.ordering
          val wrappedInterval = interval.copy(
            start = Row(interval.start),
            end = Row(interval.end))
          rangeTree.queryOverlappingValues(pkOrd, wrappedInterval).map(i => (i, r))
        } else
          Iterator()
      }

    val nParts = rvd.getNumPartitions
    val zipRDD = partitionKeyedIntervals.partitionBy(new Partitioner {
      def getPartition(key: Any): Int = key.asInstanceOf[Int]

      def numPartitions: Int = nParts
    }).values

    val localRVRowType = signature
    val pkIndex = signature.fieldIdx(key.get(0))
    val newTableType = typ.copy(rowType = newRVType)
    val newOrderedRVType = new OrderedRVDType(key.get.toArray, key.get.toArray, newRVType)
    val newRVD = leftORVD.zipPartitionsPreservesPartitioning(
      newOrderedRVType,
      zipRDD
    ) { (it, intervals) =>
      val intervalAnnotations: Array[(Interval, Any)] =
        intervals.map { r =>
          val interval = r.getAs[Interval](rightKeyFieldIdx)
          (interval, Row.fromSeq(rightValueFieldIdx.map(r.get)))
        }.toArray

      val iTree = IntervalTree.annotationTree(typOrdering, intervalAnnotations)

      val rvb = new RegionValueBuilder()
      val rv2 = RegionValue()

      it.map { rv =>
        val ur = new UnsafeRow(localRVRowType, rv)
        val pk = ur.get(pkIndex)
        val queries = iTree.queryValues(typOrdering, pk)
        val value = if (queries.isEmpty)
          null
        else
          queries(0)
        assert(typToInsert.typeCheck(value))

        rvb.set(rv.region)
        rvb.start(newRVType)

        ins(rv.region, rv.offset, rvb, () => rvb.addAnnotation(typToInsert, value))

        rv2.set(rv.region, rvb.end())

        rv2
      }
    }

    copy2(rvd = newRVD, signature = newRVType)
  }
}
