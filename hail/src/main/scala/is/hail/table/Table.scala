package is.hail.table

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.{ir, _}
import is.hail.expr.ir.{IR, LiftLiterals, TableAggregateByKey, TableExplode, TableFilter, TableIR, TableJoin, TableKeyBy, TableKeyByAndAggregate, TableLiteral, TableMapGlobals, TableMapRows, TableOrderBy, TableParallelize, TableRange, TableToMatrixTable, TableUnion, TableValue, CastTableToMatrix, _}
import is.hail.expr.types._
import is.hail.io.plink.{FamFileConfig, LoadPlink}
import is.hail.rvd._
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import is.hail.variant._
import org.apache.commons.lang3.StringUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.storage.StorageLevel
import org.json4s.jackson.JsonMethods

import scala.collection.JavaConverters._
import scala.language.implicitConversions

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
    fromDF(hc, df, key.asScala.toArray.toFastIndexedSeq)
  }

  def fromDF(hc: HailContext, df: DataFrame, key: IndexedSeq[String] = FastIndexedSeq()): Table = {
    val signature = SparkAnnotationImpex.importType(df.schema).asInstanceOf[TStruct]
    Table(hc, df.rdd, signature, key)
  }

  def read(hc: HailContext, path: String): Table =
    new Table(hc, TableIR.read(hc, path, dropRows = false, None))

  def parallelize(ir: String, nPartitions: Option[Int]): Table = {
    val rowsIR = Parser.parse_value_ir(ir)
    new Table(HailContext.get, TableParallelize(rowsIR, nPartitions))
  }

  def importFam(hc: HailContext, path: String, isQuantPheno: Boolean = false,
    delimiter: String = "\\t",
    missingValue: String = "NA"): Table = {

    val ffConfig = FamFileConfig(isQuantPheno, delimiter, missingValue)

    val (data, typ) = LoadPlink.parseFam(path, ffConfig, hc.hadoopConf)

    val rdd = hc.sc.parallelize(data)

    Table(hc, rdd, typ, IndexedSeq("id"))
  }

  def apply(
    hc: HailContext,
    rdd: RDD[Row],
    signature: TStruct
  ): Table = apply(hc, rdd, signature, FastIndexedSeq(), isSorted = false)

  def apply(
    hc: HailContext,
    rdd: RDD[Row],
    signature: TStruct,
    isSorted: Boolean
  ): Table = apply(hc, rdd, signature, FastIndexedSeq(), isSorted)

  def apply(
    hc: HailContext,
    rdd: RDD[Row],
    signature: TStruct,
    key: IndexedSeq[String]
  ): Table = apply(hc, rdd, signature, key, TStruct.empty(), Annotation.empty, isSorted = false)

  def apply(
    hc: HailContext,
    rdd: RDD[Row],
    signature: TStruct,
    key: IndexedSeq[String],
    isSorted: Boolean
  ): Table = apply(hc, rdd, signature, key, TStruct.empty(), Annotation.empty, isSorted)

  def apply(
    hc: HailContext,
    rdd: RDD[Row],
    signature: TStruct,
    key: IndexedSeq[String],
    globalSignature: TStruct,
    globals: Annotation
  ): Table = apply(
    hc,
    ContextRDD.weaken[RVDContext](rdd),
    signature,
    key,
    globalSignature,
    globals,
    isSorted = false)

  def apply(
    hc: HailContext,
    rdd: RDD[Row],
    signature: TStruct,
    key: IndexedSeq[String],
    globalSignature: TStruct,
    globals: Annotation,
    isSorted: Boolean
  ): Table = apply(
    hc,
    ContextRDD.weaken[RVDContext](rdd),
    signature,
    key,
    globalSignature,
    globals,
    isSorted)

  def apply(
    hc: HailContext,
    crdd: ContextRDD[RVDContext, Row],
    signature: TStruct,
    key: IndexedSeq[String],
    isSorted: Boolean
  ): Table = apply(hc, crdd, signature, key, TStruct.empty(), Annotation.empty, isSorted)

  def apply(
    hc: HailContext,
    crdd: ContextRDD[RVDContext, Row],
    signature: TStruct,
    key: IndexedSeq[String],
    globalSignature: TStruct,
    globals: Annotation,
    isSorted: Boolean
  ): Table = {
    val crdd2 = crdd.cmapPartitions((ctx, it) => it.toRegionValueIterator(ctx.region, signature))
    new Table(hc, TableLiteral(
      TableValue(
        TableType(signature, FastIndexedSeq(), globalSignature),
        BroadcastRow(globals.asInstanceOf[Row], globalSignature, hc.sc),
        RVD.unkeyed(signature, crdd2)))
    ).keyBy(key, isSorted)
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
    key: IndexedSeq[String] = FastIndexedSeq(),
    globalSignature: TStruct = TStruct.empty(),
    globals: Row = Row.empty
  ) = this(hc,
        TableLiteral(
          TableValue(
            TableType(signature, key, globalSignature),
            BroadcastRow(globals, globalSignature, hc.sc),
            RVD.coerce(RVDType(signature, key), crdd)))
  )

  def typ: TableType = tir.typ
  
  lazy val value: TableValue = {
    val opt = LiftLiterals(ir.Optimize(tir)).asInstanceOf[TableIR]
    opt.execute(hc)
  }

  lazy val TableValue(ktType, globals, rvd) = value

  val TableType(signature, key, globalSignature) = tir.typ

  lazy val rdd: RDD[Row] = value.rdd

  if (!(fieldNames ++ globalSignature.fieldNames).areDistinct())
    fatal(s"Column names are not distinct: ${ (fieldNames ++ globalSignature.fieldNames).duplicates().mkString(", ") }")
  if (!key.areDistinct())
    fatal(s"Key names are not distinct: ${ key.duplicates().mkString(", ") }")
  if (!key.forall(fieldNames.contains(_)))
    fatal(s"Key names found that are not column names: ${ key.filterNot(fieldNames.contains(_)).mkString(", ") }")

  def fields: Array[Field] = signature.fields.toArray

  val keyFieldIdx: Array[Int] = key.toArray.map(signature.fieldIdx)

  def keyFields: Array[Field] = key.toArray.map(signature.fieldIdx).map(i => fields(i))

  val valueFieldIdx: Array[Int] =
    signature.fields.filter(f =>
      !key.contains(f.name)
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

  def nKeys: Int = key.length

  def nPartitions: Int = rvd.getNumPartitions

  def keySignature: TStruct = tir.typ.keyType

  def valueSignature: TStruct = {
    val (t, _) = signature.filterSet(key.toSet, include = false)
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
    val fieldIndices = fields.map(f => f.name -> f.index).toMap
    val keyIndices = key.map(fieldIndices)
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
    } else if (key != other.key) {
      info(
        s"""different keys:
            | left: ${ key.map(_.mkString(", ")) }
            | right: ${ other.key.map(_.mkString(", "))}
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
    } else if (key.nonEmpty) {
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

  def aggregate(expr: String): (Any, Type) =
    aggregate(Parser.parse_value_ir(expr, IRParserEnvironment(typ.refMap)))

  def aggregate(query: IR): (Any, Type) = {
    val t = ir.TableAggregate(tir, query)
    (ir.Interpret(t, ir.Env.empty, FastIndexedSeq(), None), t.typ)
  }

  def annotateGlobal(a: Annotation, t: Type, name: String): Table = {
    new Table(hc, TableMapGlobals(tir,
      ir.InsertFields(ir.Ref("global", tir.typ.globalType), FastSeq(name -> ir.Literal.coerce(t, a)))))
  }

  def selectGlobal(expr: String): Table = {
    val ir = Parser.parse_value_ir(expr, IRParserEnvironment(typ.refMap))
    new Table(hc, TableMapGlobals(tir, ir))
  }

  def filter(cond: String, keep: Boolean): Table = {
    var irPred = Parser.parse_value_ir(cond, IRParserEnvironment(typ.refMap))
    new Table(hc,
      TableFilter(tir, ir.filterPredicateWithKeep(irPred, keep)))
  }

  def head(n: Long): Table = new Table(hc, TableHead(tir, n))

  def keyBy(keys: java.util.ArrayList[String]): Table =
    keyBy(keys.asScala.toFastIndexedSeq)

  def keyBy(keys: java.util.ArrayList[String], isSorted: Boolean): Table =
    keyBy(keys.asScala.toFastIndexedSeq, isSorted)

  def keyBy(keys: IndexedSeq[String], isSorted: Boolean = false): Table =
    new Table(hc, TableKeyBy(tir, keys, isSorted))

  def keyBy(maybeKeys: Option[IndexedSeq[String]]): Table = keyBy(maybeKeys, false)

  def keyBy(maybeKeys: Option[IndexedSeq[String]], isSorted: Boolean): Table =
    keyBy(maybeKeys.getOrElse(FastIndexedSeq()), isSorted)

  def unkey(): Table = keyBy(FastIndexedSeq())

  def mapRows(expr: String): Table =
    mapRows(Parser.parse_value_ir(expr, IRParserEnvironment(typ.refMap)))

  def mapRows(newRow: IR): Table =
    new Table(hc, TableMapRows(tir, newRow))

  def join(other: Table, joinType: String): Table =
    new Table(hc, TableJoin(this.tir, other.tir, joinType, typ.key.length))

  def leftJoinRightDistinct(other: Table, root: String): Table =
    new Table(hc, TableLeftJoinRightDistinct(tir, other.tir, root))

  def export(path: String, typesFile: String = null, header: Boolean = true, exportType: Int = ExportType.CONCATENATED) {
    ir.Interpret(ir.TableExport(tir, path, typesFile, header, exportType))
  }

  def distinctByKey(): Table = {
    new Table(hc, ir.TableDistinct(tir))
  }

  def groupByKey(name: String): Table = {
    require(key.nonEmpty)
    val sorted = keyBy(key.toArray)
    sorted.copy2(
      rvd = sorted.rvd.groupByKey(name),
      signature = keySignature ++ TStruct(name -> TArray(valueSignature)))
  }

  def jToMatrixTable(rowKeys: java.util.ArrayList[String],
    colKeys: java.util.ArrayList[String],
    rowFields: java.util.ArrayList[String],
    colFields: java.util.ArrayList[String],
    nPartitions: java.lang.Integer): MatrixTable = {

    toMatrixTable(rowKeys.asScala.toArray, colKeys.asScala.toArray,
      rowFields.asScala.toArray, colFields.asScala.toArray,
      Option(nPartitions).map(_.asInstanceOf[Int])
    )
  }

  def toMatrixTable(
    rowKeys: Array[String],
    colKeys: Array[String],
    rowFields: Array[String],
    colFields: Array[String],
    nPartitions: Option[Int] = None
  ): MatrixTable = {
    new MatrixTable(hc, TableToMatrixTable(tir, rowKeys, colKeys, rowFields, colFields, nPartitions))
  }

  def keyByAndAggregate(expr: String, key: String, nPartitions: Option[Int], bufferSize: Int): Table = {
    new Table(hc, TableKeyByAndAggregate(tir,
      Parser.parse_value_ir(expr, IRParserEnvironment(typ.refMap)),
      Parser.parse_value_ir(key, IRParserEnvironment(typ.refMap)),
      nPartitions,
      bufferSize
    ))
  }

  def unlocalizeEntries(
    entriesFieldName: String,
    colsFieldName: String,
    colKey: java.util.ArrayList[String]
  ): MatrixTable =
    new MatrixTable(hc,
      CastTableToMatrix(tir, entriesFieldName, colsFieldName, colKey.asScala.toFastIndexedSeq))

  def aggregateByKey(expr: String): Table = {
    val x = Parser.parse_value_ir(expr, IRParserEnvironment(typ.refMap))
    new Table(hc, TableAggregateByKey(tir, x))
  }

  def expandTypes(): Table = {
    require(typ.key.isEmpty)

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
    val newRVD = rvd.truncateKey(IndexedSeq())
    copy2(
      rvd = newRVD.changeType(newRowType),
      signature = newRowType)
  }

  def flatten(): Table = {
    require(typ.key.isEmpty)

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

    new Table(hc, ir.TableMapRows(tir, ir.MakeStruct(newFields.flatten)))
  }

  // expandTypes must be called before toDF
  def toDF(sqlContext: SQLContext): DataFrame = {
    val localSignature = signature.physicalType
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


  def write(path: String, overwrite: Boolean = false, stageLocally: Boolean = false, codecSpecJSONStr: String = null) {
    ir.Interpret(ir.TableWrite(tir, path, overwrite, stageLocally, codecSpecJSONStr))
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
    new Table(hc, TableOrderBy(TableKeyBy(tir, FastIndexedSeq()), sortFields))
  }

  def rename(rowMap: java.util.HashMap[String, String], globalMap: java.util.HashMap[String, String]): Table =
    new Table(hc, TableRename(tir, rowMap.asScala.toMap, globalMap.asScala.toMap))

  def repartition(n: Int, shuffle: Boolean = true): Table = new Table(hc, ir.TableRepartition(tir, n, shuffle))

  def union(kts: java.util.ArrayList[Table]): Table = union(kts.asScala.toArray: _*)

  def union(kts: Table*): Table = new Table(hc, TableUnion((tir +: kts.map(_.tir)).toFastIndexedSeq))

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
      val takeResult = head(n + 1).collect()
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

    val sb = new StringBuilder()
    val config = ShowStrConfig()
    def convertValue(t: Type, v: Annotation, ab: ArrayBuilder[String]) {
      t match {
        case s: TStruct =>
          val r = v.asInstanceOf[Row]
          s.fields.foreach(f => convertValue(f.typ, if (r == null) null else r.get(f.index), ab))
        case _ =>
          ab += t.showStr(v, config, sb)
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

  def copy2(rvd: RVD = rvd,
    signature: TStruct = signature,
    key: IndexedSeq[String] = key,
    globalSignature: TStruct = globalSignature,
    globals: BroadcastRow = globals): Table = {
    new Table(hc, TableLiteral(
      TableValue(TableType(signature, key, globalSignature), globals, rvd)
    ))
  }

  def intervalJoin(other: Table, fieldName: String): Table = {
    assert(other.keySignature.size == 1 && other.keySignature.types(0).isInstanceOf[TInterval])
    val intervalType = other.keySignature.types(0).asInstanceOf[TInterval]
    assert(keySignature.size == 1 && keySignature.types(0) == intervalType.pointType)

    val typToInsert: Type = other.valueSignature
    val (newRowPType, ins) = signature.physicalType.unsafeStructInsert(typToInsert.physicalType, List(fieldName))
    val newRowType = newRowPType.virtualType

    val rightTyp = other.typ
    val leftRVDType = typ.rvdType

    val zipper = { (ctx: RVDContext, it: Iterator[RegionValue], intervals: Iterator[RegionValue]) =>
      val rvb = new RegionValueBuilder()
      val rv2 = RegionValue()
      OrderedRVIterator(leftRVDType, it, ctx).leftIntervalJoinDistinct(
        OrderedRVIterator(rightTyp.rvdType, intervals, ctx)
      )
        .map { case Muple(rv, i) =>
          rvb.set(rv.region)
          rvb.start(newRowType)
          ins(
            rv.region,
            rv.offset,
            rvb,
            () => if (i == null) rvb.setMissing() else rvb.selectRegionValue(rightTyp.rowType, rightTyp.valueFieldIdx, i))
          rv2.set(rv.region, rvb.end())

          rv2
        }
    }

    val newRVDType = RVDType(newRowType, key)
    val newRVD = rvd.intervalAlignAndZipPartitions(newRVDType, other.rvd)(zipper)

    copy2(rvd = newRVD, signature = newRowType)
  }

  def filterPartitions(parts: java.util.ArrayList[Int], keep: Boolean): Table =
    filterPartitions(parts.asScala.toArray, keep)

  def filterPartitions(parts: Array[Int], keep: Boolean = true): Table = {
    copy2(rvd =
      rvd.subsetPartitions(
        if (keep)
          parts
        else {
          val partSet = parts.toSet
          (0 until rvd.getNumPartitions).filter(i => !partSet.contains(i)).toArray
        })
    )
  }
}
