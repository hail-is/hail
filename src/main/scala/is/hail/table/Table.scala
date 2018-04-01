package is.hail.table

import is.hail.HailContext
import is.hail.annotations._
import is.hail.annotations.aggregators.RegionValueAggregator
import is.hail.expr._
import is.hail.expr.ir.{MakeTuple, CompileWithAggregators}
import is.hail.expr.types._
import is.hail.io.annotators.{BedAnnotator, IntervalList}
import is.hail.io.plink.{FamFileConfig, LoadPlink}
import is.hail.io.{CassandraConnector, CodecSpec, SolrConnector, exportTypes}
import is.hail.methods.Aggregators
import is.hail.rvd._
import is.hail.utils._
import is.hail.variant._
import org.apache.commons.lang3.StringUtils
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

object SortColumn {
  implicit def fromColumn(column: String): SortColumn = SortColumn(column, Ascending)
}

case class SortColumn(column: String, sortOrder: SortOrder)

case class TableSpec(
  file_version: Int,
  hail_version: String,
  references_rel_path: String,
  table_type: TableType,
  components: Map[String, ComponentSpec]) extends RelationalSpec {
  def rowsComponent: RVDComponentSpec = getComponent[RVDComponentSpec]("rows")
}

object Table {
  def range(hc: HailContext, n: Int, name: String = "index", partitions: Option[Int] = None): Table = {
    val range = Range(0, n).view.map(Row(_))
    val rdd = partitions match {
      case Some(parts) => hc.sc.parallelize(range, numSlices = parts)
      case None => hc.sc.parallelize(range)
    }
    Table(hc, rdd, TStruct(name -> TInt32()), IndexedSeq(name))
  }

  def fromDF(hc: HailContext, df: DataFrame, key: java.util.ArrayList[String]): Table = {
    fromDF(hc, df, key.asScala.toArray.toFastIndexedSeq)
  }

  def fromDF(hc: HailContext, df: DataFrame, key: IndexedSeq[String] = Array.empty[String]): Table = {
    val signature = SparkAnnotationImpex.importType(df.schema).asInstanceOf[TStruct]
    Table(hc, df.rdd.map { r =>
      SparkAnnotationImpex.importAnnotation(r, signature).asInstanceOf[Row]
    },
      signature, key)
  }

  def read(hc: HailContext, path: String): Table = {

    val spec = (RelationalSpec.read(hc, path): @unchecked) match {
      case ts: TableSpec => ts
      case _: MatrixTableSpec => fatal(s"file is a MatrixTable, not a Table: '$path'")
    }

    val successFile = path + "/_SUCCESS"
    if (!hc.hadoopConf.exists(path + "/_SUCCESS"))
      fatal(s"write failed: file not found: $successFile")
    new Table(hc, TableRead(path, spec, dropRows = false))
  }

  def parallelize(hc: HailContext, rowsJSON: String, signature: TStruct,
    keyNames: java.util.ArrayList[String], nPartitions: Option[Int]): Table = {
    parallelize(hc, rowsJSON, signature, keyNames.asScala.toArray, nPartitions)
  }

  def parallelize(hc: HailContext, rowsJSON: String, signature: TStruct,
    key: IndexedSeq[String], nPartitions: Option[Int] = None): Table = {
    val typ = TableType(signature, key.toArray.toFastIndexedSeq, TStruct())
    val parsedRows = JSONAnnotationImpex.importAnnotation(JsonMethods.parse(rowsJSON), TArray(signature))
    new Table(hc, TableParallelize(typ, parsedRows.asInstanceOf[IndexedSeq[Row]], nPartitions))
  }

  def importIntervalList(hc: HailContext, filename: String,
    rg: Option[ReferenceGenome] = Some(ReferenceGenome.defaultReference),
    skipInvalidIntervals: Boolean = false): Table = {
    IntervalList.read(hc, filename, rg, skipInvalidIntervals)
  }

  def importBED(hc: HailContext, filename: String,
    rg: Option[ReferenceGenome] = Some(ReferenceGenome.defaultReference),
    skipInvalidIntervals: Boolean = false): Table = {
    BedAnnotator.apply(hc, filename, rg, skipInvalidIntervals)
  }

  def importFam(hc: HailContext, path: String, isQuantPheno: Boolean = false,
    delimiter: String = "\\t",
    missingValue: String = "NA"): Table = {

    val ffConfig = FamFileConfig(isQuantPheno, delimiter, missingValue)

    val (data, typ) = LoadPlink.parseFam(path, ffConfig, hc.hadoopConf)

    val rdd = hc.sc.parallelize(data)

    Table(hc, rdd, typ, Array("id"))
  }

  def apply(hc: HailContext, rdd: RDD[Row], signature: TStruct, key: IndexedSeq[String] = Array.empty[String],
    globalSignature: TStruct = TStruct.empty(), globals: Annotation = Annotation.empty): Table = {
    val rdd2 = rdd.mapPartitions(_.toRegionValueIterator(signature))
    new Table(hc, TableLiteral(
      TableValue(TableType(signature, key, globalSignature),
        BroadcastValue(globals, globalSignature, hc.sc),
        new UnpartitionedRVD(signature, rdd2))
    ))
  }

  def sameWithinTolerance(t: Type, l: Array[Row], r: Array[Row], tolerance: Double): Boolean = {
    val used = new Array[Boolean](r.length)
    var i = 0
    while (i < l.length) {
      val li = l(i)
      var matched = false
      var j = 0
      while (!matched && j < l.length && !used(j)) {
        matched = t.valuesSimilar(li, r(j), tolerance)
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

  def this(hc: HailContext,
    rdd: RDD[RegionValue],
    signature: TStruct,
    key: IndexedSeq[String] = Array.empty[String],
    globalSignature: TStruct = TStruct.empty(),
    globals: Row = Row.empty) = {
    this(hc, TableLiteral(
      TableValue(TableType(signature, key, globalSignature), BroadcastValue(globals, globalSignature, hc.sc),
        new UnpartitionedRVD(signature, rdd))
    ))
  }

  lazy val value: TableValue = {
    val opt = TableIR.optimize(tir)
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

  def rowEvalContext(): EvalContext = {
    val ec = EvalContext(
      "global" -> globalSignature,
      "row" -> signature)
    ec
  }

  private def aggEvalContext(): EvalContext = {
    val aggSymbolTable = Map(
      "global" -> (0, globalSignature),
      "row" -> (1, signature)
    )
    val ec = EvalContext("global" -> globalSignature,
      "AGG" -> TAggregable(signature, aggSymbolTable))
    ec
  }

  def fields: Array[Field] = signature.fields.toArray

  val keyFieldIdx: Array[Int] = key.toArray.map(signature.fieldIdx)

  def keyFields: Array[Field] = key.toArray.map(signature.fieldIdx).map(i => fields(i))

  val valueFieldIdx: Array[Int] = signature.fields.filter(f => !key.contains(f.name)).map(_.index).toArray

  def fieldNames: Array[String] = fields.map(_.name)

  def partitionCounts(): Array[Long] = {
    tir.partitionCounts match {
      case Some(counts) => counts
      case None => rvd.countPerPartition()
    }
  }

  def count(): Long = partitionCounts().sum

  def forceCount(): Long = rvd.count()

  def nColumns: Int = fields.length

  def nKeys: Int = key.length

  def nPartitions: Int = rvd.partitions.length

  def keySignature: TStruct = {
    val (t, _) = signature.select(key.toArray)
    t
  }

  def valueSignature: TStruct = {
    val (t, _) = signature.filter(key.toSet, include = false)
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

  def same(other: Table, tolerance: Double = defaultTolerance): Boolean = {
    val localValueSignature = valueSignature

    val globalSignatureOpt = globalSignature.deepOptional()
    if (signature.deepOptional() != other.signature.deepOptional()) {
      info(
        s"""different signatures:
           | left: ${ signature.toString }
           | right: ${ other.signature.toString }
           |""".stripMargin)
      false
    } else if (key.toSeq != other.key.toSeq) {
      info(
        s"""different key names:
           | left: ${ key.mkString(", ") }
           | right: ${ other.key.mkString(", ") }
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
    } else {
      keyedRDD().groupByKey().fullOuterJoin(other.keyedRDD().groupByKey()).forall { case (k, (v1, v2)) =>
        (v1, v2) match {
          case (None, None) => true
          case (Some(x), Some(y)) =>
            val r1 = x.toArray
            val r2 = y.toArray
            val res = if (r1.length != r2.length)
              false
            else r1.counter() == r2.counter() ||
              Table.sameWithinTolerance(localValueSignature, r1, r2, tolerance)
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

  def queryJSON(expr: String): String = {
    val (a, t) = query(expr)
    val jv = JSONAnnotationImpex.exportAnnotation(a, t)
    JsonMethods.compact(jv)
  }

  def query(expr: String): (Annotation, Type) = query(Array(expr)).head

  def query(exprs: Array[String]): Array[(Annotation, Type)] = {
    val globalsBc = globals.broadcast
    val ec = aggEvalContext()
    val irs = exprs.flatMap(Parser.parseToAST(_, ec).toIR(Some("AGG")))

    if (irs.length == exprs.length) {
      val ir = MakeTuple(irs)

      val localGlobalSignature = globalSignature
      val tAgg = ec.st("AGG")._2.asInstanceOf[TAggregable]

      val (rvAggs, seqOps, aggResultType, f, t) = CompileWithAggregators[Long, Long, Long, Long, Long](
        "AGG", tAgg,
        "global", globalSignature,
        ir)

      val aggResults = if (seqOps.nonEmpty) {
        rvd.treeAggregate[Array[RegionValueAggregator]](rvAggs)({ case (rvaggs, rv) =>
          // add globals to region value
          val rowOffset = rv.offset
          val rvb = new RegionValueBuilder()
          rvb.set(rv.region)
          rvb.start(localGlobalSignature)
          rvb.addAnnotation(localGlobalSignature, globalsBc.value)
          val globalsOffset = rvb.end()

          rvaggs.zip(seqOps).foreach { case (rvagg, seqOp) =>
            seqOp()(rv.region, rvagg, rowOffset, false, globalsOffset, false, rowOffset, false)
          }
          rvaggs
        }, { (rvAggs1, rvAggs2) =>
          rvAggs1.zip(rvAggs2).foreach { case (rvAgg1, rvAgg2) => rvAgg1.combOp(rvAgg2) }
          rvAggs1
        })
      } else
        Array.empty[RegionValueAggregator]

      val region: Region = Region()
      val rvb: RegionValueBuilder = new RegionValueBuilder()
      rvb.set(region)

      rvb.start(aggResultType)
      rvb.startStruct()
      aggResults.foreach(_.result(rvb))
      rvb.endStruct()
      val aggResultsOffset = rvb.end()

      rvb.start(globalSignature)
      rvb.addAnnotation(globalSignature, globalsBc.value)
      val globalsOffset = rvb.end()

      val resultOffset = f()(region, aggResultsOffset, false, globalsOffset, false)
      val resultType = coerce[TTuple](t)
      val result = UnsafeRow.readBaseStruct(resultType, region, resultOffset)

      result.toSeq.zip(resultType.types).toArray
    } else {
      val ts = exprs.map(e => Parser.parseExpr(e, ec))

      val (zVals, seqOp, combOp, resultOp) = Aggregators.makeFunctions[Annotation](ec, {
        case (ec, a) =>
          ec.setAll(globalsBc.value, a)
      })

      val r = rdd.aggregate(zVals.map(_.copy()))(seqOp, combOp)
      resultOp(r)

      ts.map { case (t, f) => (f(), t) }
    }
  }

  def annotateGlobal(a: Annotation, t: Type, name: String): Table = {
    val (newT, i) = globalSignature.insert(t, name)
    copy2(globalSignature = newT.asInstanceOf[TStruct],
      globals = globals.copy(value = i(globals.value, a), t = newT))
  }

  def annotateGlobalJSON(s: String, t: Type, name: String): Table = {
    val ann = JSONAnnotationImpex.importAnnotation(JsonMethods.parse(s), t)

    annotateGlobal(ann, t, name)
  }

  def selectGlobal(expr: String): Table = {
    val ec = EvalContext("global" -> globalSignature)
    ec.set(0, globals.value)

    val ast = Parser.parseToAST(expr, ec)
    assert(ast.`type`.isInstanceOf[TStruct])

    ast.toIR() match {
      case Some(ir) if ast.`type`.asInstanceOf[TStruct].size < 500 =>
        new Table(hc, TableMapGlobals(tir, ir))
      case _ =>
        val (t, f) = Parser.parseExpr(expr, ec)
        val newSignature = t.asInstanceOf[TStruct]
        val newGlobal = f()

        copy2(globalSignature = newSignature,
          globals = globals.copy(value = newGlobal, t = newSignature))
    }
  }

  def filter(cond: String, keep: Boolean): Table = {
    val ec = rowEvalContext()
    var filterAST = Parser.parseToAST(cond, ec)
    val pred = filterAST.toIR()
    pred match {
      case Some(irPred) =>
        new Table(hc,
          TableFilter(tir, ir.filterPredicateWithKeep(irPred, keep, "filter_pred"))
        )
      case None =>
        if (!keep)
          filterAST = Apply(filterAST.getPos, "!", Array(filterAST))
        val f: () => java.lang.Boolean = Parser.evalTypedExpr[java.lang.Boolean](filterAST, ec)
        val localSignature = signature

        val globalsBc = globals.broadcast
        val p = (rv: RegionValue) => {
          val ur = new UnsafeRow(localSignature, rv)
          ec.setAll(globalsBc.value, ur)
          val ret = f()
          ret != null && ret.booleanValue()
        }
        copy2(rvd = rvd.filter(p))
    }
  }

  def head(n: Long): Table = {
    if (n < 0)
      fatal(s"n must be non-negative! Found `$n'.")
    copy(rdd = rdd.head(n))
  }

  def keyBy(key: String*): Table = keyBy(key)

  def keyBy(key: java.util.ArrayList[String]): Table = keyBy(key.asScala)

  def keyBy(key: Iterable[String]): Table = {
    val colSet = fieldNames.toSet
    val badKeys = key.filter(!colSet.contains(_))

    if (badKeys.nonEmpty)
      fatal(
        s"""Invalid ${ plural(badKeys.size, "key") }: [ ${ badKeys.map(x => s"'$x'").mkString(", ") } ]
           |  Available columns: [ ${ signature.fields.map(x => s"'${ x.name }'").mkString(", ") } ]""".stripMargin)

    copy(key = key.toArray[String])
  }

  def select(expr: String): Table = {
    val ec = rowEvalContext()
    val ast = Parser.parseToAST(expr, ec)
    assert(ast.`type`.isInstanceOf[TStruct])

    ast.toIR() match {
      case Some(ir) if ast.`type`.asInstanceOf[TStruct].size < 500 =>
        new Table(hc, TableMapRows(tir, ir))
      case _ =>
        val (t, f) = Parser.parseExpr(expr, ec)
        val newSignature = t.asInstanceOf[TStruct]
        val globalsBc = globals.broadcast

        val annotF: Row => Row = { r =>
          ec.setAll(globalsBc.value, r)
          f().asInstanceOf[Row]
        }

        val newKey = key.filter(newSignature.fieldNames.toSet)

        copy(rdd = rdd.map(annotF), signature = newSignature, key = newKey)
    }
  }

  def join(other: Table, joinType: String): Table =
    new Table(hc, TableJoin(this.tir, other.tir, joinType))

  def export(output: String, typesFile: String = null, header: Boolean = true, exportType: Int = ExportType.CONCATENATED) {
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
    }.writeTable(output, hc.tmpDir, Some(fields.map(_.name).mkString("\t")).filter(_ => header), exportType = exportType)
  }

  def distinctByKey(): Table = {
    copy2(rvd = toOrderedRVD(hintPartitioner = None, partitionKeys = key.length).distinctByKey())
  }

  def groupByKey(name: String): Table = {
    copy2(rvd = toOrderedRVD(hintPartitioner = None, partitionKeys = key.length).groupByKey(name),
      signature = keySignature ++ TStruct(name -> TArray(valueSignature)))
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
      Option(nPartitions)
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

    // all keys accounted for
    assert(rowKeys.length + colKeys.length == key.length)
    assert(rowKeys.toSet.union(colKeys.toSet) == key.toSet)

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
        val rvCopy = rv.copy()
        ur.set(rvCopy)

        val colKey = Row.fromSeq(colKeyIndices.map(ur.get))
        val colValues = Row.fromSeq(colValueIndices.map(ur.get))
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

    val newRVD = ordered.mapPartitionsPreservesPartitioning(matrixType.orvdType) { it =>
      val region = Region()
      val rvb = new RegionValueBuilder(region)
      val outRV = RegionValue(region)

      OrderedRVIterator(
        new OrderedRVDType(partitionKeys, rowKeys, rowEntryStruct),
        it
      ).staircase.map { rowIt =>
        region.clear()
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
    }
    new MatrixTable(hc,
      matrixType,
      globals,
      colDataConcat,
      newRVD)
  }

  def aggregate(keyCond: String, aggCond: String, nPartitions: Option[Int] = None): Table = {

    val ec = aggEvalContext()
    val keyEC = rowEvalContext()

    val (keyPaths, keyTypes, keyF) = Parser.parseAnnotationExprs(keyCond, keyEC, None)

    val (aggPaths, aggTypes, aggF) = Parser.parseAnnotationExprs(aggCond, ec, None)

    val newKey = keyPaths.map(_.head)
    val aggNames = aggPaths.map(_.head)

    val keySignature = TStruct((newKey, keyTypes).zipped.toSeq: _*)
    val aggSignature = TStruct((aggNames, aggTypes).zipped.toSeq: _*)
    val globalsBc = globals.broadcast

    // FIXME: delete this when we understand what it's doing
    ec.set(0, globals.safeValue)
    val (zVals, seqOp, combOp, resultOp) = Aggregators.makeFunctions[Row](ec, {
      case (ec_, r) =>
        ec_.set(0, globalsBc.value)
        ec_.set(1, r)
    })

    val newRDD = rdd.mapPartitions {
      it =>
        it.map {
          r =>
            keyEC.set(0, globalsBc.value)
            keyEC.set(1, r)
            val key = Row.fromSeq(keyF())
            (key, r)
        }
    }.aggregateByKey(zVals, nPartitions.getOrElse(this.nPartitions))(seqOp, combOp)
      .map {
        case (k, agg) =>
          ec.set(0, globalsBc.value)
          resultOp(agg)
          Row.fromSeq(k.toSeq ++ aggF())
      }

    copy(rdd = newRDD, signature = keySignature.merge(aggSignature)._1, key = newKey)
  }

  def expandTypes(): Table = {
    val localSignature = signature
    val expandedSignature = Annotation.expandType(localSignature).asInstanceOf[TStruct]

    copy(rdd = rdd.map { a => Annotation.expandAnnotation(a, localSignature).asInstanceOf[Row] },
      signature = expandedSignature,
      key = key)
  }

  def flatten(): Table = {
    val localSignature = signature
    val keySignature = TStruct(keyFields.map { f => f.name -> f.typ }: _*)
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

  def explode(columnToExplode: String): Table = {

    val explodeField = signature.fieldOption(columnToExplode) match {
      case Some(x) => x
      case None =>
        fatal(
          s"""Input field name `${ columnToExplode }' not found in Table.
             |Table field names are `${ fieldNames.mkString(", ") }'.""".stripMargin)
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
    val codecSpec =
      if (codecSpecJSONStr != null) {
        implicit val formats = RVDSpec.formats
        val codecSpecJSON = JsonMethods.parse(codecSpecJSONStr)
        codecSpecJSON.extract[CodecSpec]
      } else
        CodecSpec.default

    if (overwrite)
      hc.hadoopConf.delete(path, recursive = true)
    else if (hc.hadoopConf.exists(path))
      fatal(s"file already exists: $path")

    hc.hadoopConf.mkDir(path)

    val globalsPath = path + "/globals"
    hc.hadoopConf.mkDir(globalsPath)
    RVD.writeLocalUnpartitioned(hc, globalsPath, globalSignature, codecSpec, Array(globals.value))

    val partitionCounts = rvd.write(path + "/rows", codecSpec)

    val referencesPath = path + "/references"
    hc.hadoopConf.mkDir(referencesPath)
    ReferenceGenome.exportReferences(hc, referencesPath, signature)
    ReferenceGenome.exportReferences(hc, referencesPath, globalSignature)

    val spec = TableSpec(
      FileFormat.version.rep,
      hc.version,
      "references",
      tir.typ,
      Map("globals" -> RVDComponentSpec("globals"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    spec.write(hc, path)

    hc.hadoopConf.writeTextFile(path + "/_SUCCESS")(out => ())
  }

  def cache(): Table = persist("MEMORY_ONLY")

  def persist(storageLevel: String): Table = {
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

  def orderBy(sortCols: SortColumn*): Table =
    orderBy(sortCols.toArray)

  def orderBy(sortCols: Array[SortColumn]): Table = {
    val sortColIndexOrd = sortCols.map { case SortColumn(n, so) =>
      val i = signature.fieldIdx(n)
      val f = signature.fields(i)
      val fo = f.typ.ordering
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

  def repartition(n: Int, shuffle: Boolean = true): Table = copy(rdd = rdd.coalesce(n, shuffle))

  def union(kts: java.util.ArrayList[Table]): Table = union(kts.asScala.toArray: _*)

  def union(kts: Table*): Table = {
    kts.foreach { kt =>
      if (signature != kt.signature)
        fatal("cannot union tables with different schemas")
      if (!key.sameElements(kt.key))
        fatal("cannot union tables with different key")
    }

    copy(rdd = hc.sc.union(rdd, kts.map(_.rdd): _*))
  }

  def take(n: Int): Array[Row] = rdd.take(n)

  def takeJSON(n: Int): String = {
    val r = JSONAnnotationImpex.exportAnnotation(take(n).toFastIndexedSeq, TArray(signature))
    JsonMethods.compact(r)
  }

  def sample(p: Double, seed: Int = 1): Table = {
    require(p > 0 && p < 1, s"the 'p' parameter must fall between 0 and 1, found $p")
    copy2(rvd = rvd.sample(withReplacement = false, p, seed))
  }

  def index(name: String = "index"): Table = {
    if (fieldNames.contains(name))
      fatal(s"name collision: cannot index table, because column '$name' already exists")

    val (newSignature, ins) = signature.insert(TInt64(), name)

    val newRDD = rdd.zipWithIndex().map { case (r, ind) => ins(r, ind).asInstanceOf[Row] }

    copy(signature = newSignature.asInstanceOf[TStruct], rdd = newRDD)
  }

  def maximalIndependentSet(iExpr: String, jExpr: String, tieBreakerExpr: Option[String]): Array[Any] = {
    val ec = rowEvalContext()

    val (iType, iThunk) = Parser.parseExpr(iExpr, ec)
    val (jType, jThunk) = Parser.parseExpr(jExpr, ec)

    if (iType != jType)
      fatal(s"node expressions must have the same type: type of `i' is $iType, but type of `j' is $jType")

    val tieBreakerEc = EvalContext("l" -> iType, "r" -> iType)

    val maybeTieBreaker = tieBreakerExpr.map { e =>
      val tieBreakerThunk = Parser.parseTypedExpr[Long](e, tieBreakerEc)

      (l: Any, r: Any) => {
        tieBreakerEc.setAll(l, r)
        tieBreakerThunk()
      }
    }.getOrElse(null)

    val globalsBc = globals.broadcast

    val edgeRdd = rdd.map { r =>
      ec.setAll(globalsBc.value, r)
      (iThunk(), jThunk())
    }

    if (edgeRdd.count() > 400000)
      warn(s"over 400,000 edges are in the graph; maximal_independent_set may run out of memory")

    Graph.maximalIndependentSet(edgeRdd.collect(), maybeTieBreaker)
  }

  def maximalIndependentSet(iExpr: String, jExpr: String, keep: Boolean,
    maybeTieBreaker: Option[String] = None): Table = {

    val (iType, _) = Parser.parseExpr(iExpr, rowEvalContext())

    val relatedNodesToKeep = this.maximalIndependentSet(iExpr, jExpr, maybeTieBreaker).toSet

    val nodes = this.select(s"{node : [$iExpr, $jExpr] }")
      .explode("node")
      .keyBy("node")

    nodes.annotateGlobal(relatedNodesToKeep, TSet(iType), "relatedNodesToKeep")
      .filter(s"global.relatedNodesToKeep.contains(row.node)", keep = keep)
      .selectGlobal("{}")
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
    key: IndexedSeq[String] = key,
    globalSignature: TStruct = globalSignature,
    newGlobals: Annotation = globals.value): Table = {
    Table(hc, rdd, signature, key, globalSignature, newGlobals)
  }

  def copy2(rvd: RVD = rvd,
    signature: TStruct = signature,
    key: IndexedSeq[String] = key,
    globalSignature: TStruct = globalSignature,
    globals: BroadcastValue = globals): Table = {
    new Table(hc, TableLiteral(
      TableValue(TableType(signature, key, globalSignature), globals, rvd)
    ))
  }

  def toOrderedRVD(hintPartitioner: Option[OrderedRVDPartitioner], partitionKeys: Int): OrderedRVD = {
    val orderedKTType = new OrderedRVDType(key.take(partitionKeys).toArray, key.toArray, signature)
    assert(hintPartitioner.forall(p => p.pkType.types.sameElements(orderedKTType.pkType.types)))
    OrderedRVD.coerce(orderedKTType, rvd, None, hintPartitioner)
  }
}
