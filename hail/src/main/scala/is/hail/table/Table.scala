package is.hail.table

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.ir._
import is.hail.expr.types._
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.expr.{ir, _}
import is.hail.io.plink.{FamFileConfig, LoadPlink}
import is.hail.rvd._
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.json4s.jackson.JsonMethods

import scala.collection.JavaConverters._
import scala.language.implicitConversions

sealed abstract class SortOrder

case object Ascending extends SortOrder

case object Descending extends SortOrder

case class SortField(field: String, sortOrder: SortOrder)

abstract class AbstractTableSpec extends RelationalSpec {
  def references_rel_path: String
  def table_type: TableType
  def rowsComponent: RVDComponentSpec = getComponent[RVDComponentSpec]("rows")

  def rowsSpec(path: String): AbstractRVDSpec = AbstractRVDSpec.read(HailContext.get, path + "/" + rowsComponent.rel_path)
  def indexed(path: String): Boolean = rowsSpec(path).indexed
}

case class TableSpec(
  file_version: Int,
  hail_version: String,
  references_rel_path: String,
  table_type: TableType,
  components: Map[String, ComponentSpec]) extends AbstractTableSpec

object Table {
  def range(hc: HailContext, n: Int, nPartitions: Option[Int] = None): Table =
    new Table(hc, TableRange(n, nPartitions.getOrElse(hc.sc.defaultParallelism)))

  def pyFromDF(df: DataFrame, jKey: java.util.List[String]): TableIR = {
    val key = jKey.asScala.toArray.toFastIndexedSeq
    val signature = SparkAnnotationImpex.importType(df.schema).asInstanceOf[TStruct]
    ExecuteContext.scoped { ctx =>
      TableLiteral(TableValue(ctx, signature, key, df.rdd), ctx)
    }
  }

  def read(hc: HailContext, path: String): Table =
    new Table(hc, TableIR.read(hc, path, dropRows = false, None))

  def importFamJSON(path: String, isQuantPheno: Boolean = false,
    delimiter: String = "\\t",
    missingValue: String = "NA"): String = {
    val ffConfig = FamFileConfig(isQuantPheno, delimiter, missingValue)
    val (data, ptyp) = LoadPlink.parseFam(path, ffConfig, HailContext.sFS)
    val jv = JSONAnnotationImpex.exportAnnotation(
      Row(ptyp.toString, data),
      TStruct("type" -> TString(), "data" -> TArray(ptyp.virtualType)))
    JsonMethods.compact(jv)
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
    val pType = PType.canonical(signature).asInstanceOf[PStruct]
    val crdd2 = crdd.cmapPartitions((ctx, it) => it.toRegionValueIterator(ctx.region, pType))
    ExecuteContext.scoped { ctx =>
      new Table(hc, TableKeyBy(TableLiteral(
        TableValue(
          TableType(signature, FastIndexedSeq(), globalSignature),
          BroadcastRow(ctx, globals.asInstanceOf[Row], globalSignature),
          RVD.unkeyed(pType, crdd2)), ctx),
        key, isSorted))
    }
  }

  def apply(
    hc: HailContext,
    rdd: RDD[Row],
    signature: PStruct,
    key: IndexedSeq[String]
  ): Table = apply(hc, ContextRDD.weaken[RVDContext](rdd), signature, key, TStruct.empty(), Annotation.empty, false)

  def apply(
    hc: HailContext,
    crdd: ContextRDD[RVDContext, Row],
    signature: PStruct,
    key: IndexedSeq[String],
    globalSignature: TStruct,
    globals: Annotation,
    isSorted: Boolean
  ): Table = {
    val crdd2 = crdd.cmapPartitions((ctx, it) => it.toRegionValueIterator(ctx.region, signature))
    ExecuteContext.scoped { ctx =>
      new Table(hc, TableLiteral(
        TableValue(
          TableType(signature.virtualType, FastIndexedSeq(), globalSignature),
          BroadcastRow(ctx, globals.asInstanceOf[Row], globalSignature),
          RVD.unkeyed(signature, crdd2)),
        ctx)
      ).keyBy(key, isSorted)
    }
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
    true
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
  ) = this(hc, ExecuteContext.scoped { ctx =>
        TableLiteral(
          TableValue(
            TableType(signature, key, globalSignature),
            BroadcastRow(ctx, globals, globalSignature),
            RVD.coerce(RVDType(PType.canonical(signature).asInstanceOf[PStruct], key), crdd)), ctx)}
  )

  def typ: TableType = tir.typ

  lazy val (globals, rvd, rdd, lit) = {
    ExecuteContext.scoped { ctx =>
      val tv = Interpret(tir, ctx, optimize = true)
      (tv.globals.safeJavaValue, tv.rvd, tv.rdd, TableLiteral(tv, ctx))
    }
  }

  val TableType(signature, key, globalSignature) = tir.typ

  if (!(fieldNames ++ globalSignature.fieldNames).areDistinct())
    fatal(s"Column names are not distinct: ${ (fieldNames ++ globalSignature.fieldNames).duplicates().mkString(", ") }")
  if (!key.areDistinct())
    fatal(s"Key names are not distinct: ${ key.duplicates().mkString(", ") }")
  if (!key.forall(fieldNames.contains(_)))
    fatal(s"Key names found that are not column names: ${ key.filterNot(fieldNames.contains(_)).mkString(", ") }")

  def fields: Array[Field] = signature.fields.toArray

  def fieldNames: Array[String] = fields.map(_.name)

  def nPartitions: Int = rvd.getNumPartitions

  def count(): Long = ExecuteContext.scoped { ctx => ir.Interpret[Long](ctx, ir.TableCount(tir)) }

  def valueSignature: TStruct = {
    val (t, _) = signature.filterSet(key.toSet, include = false)
    t
  }

  def typeCheck() {

    if (!globalSignature.typeCheck(globals)) {
      fatal(
        s"""found violation of global signature
           |  Schema: ${ globalSignature.toString }
           |  Annotation: ${ globals }""".stripMargin)
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
    } else if (!globalSignatureOpt.valuesSimilar(globals, other.globals)) {
      info(
        s"""different global annotations:
           | left: ${ globals }
           | right: ${ other.globals }
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

  def annotateGlobal(a: Annotation, t: Type, name: String): Table = {
    new Table(hc, TableMapGlobals(tir,
      ir.InsertFields(ir.Ref("global", tir.typ.globalType), FastSeq(name -> ir.Literal.coerce(t, a)))))
  }

  def keyBy(keys: IndexedSeq[String], isSorted: Boolean = false): Table =
    new Table(hc, TableKeyBy(tir, keys, isSorted))

  def keyBy(maybeKeys: Option[IndexedSeq[String]]): Table = keyBy(maybeKeys, false)

  def keyBy(maybeKeys: Option[IndexedSeq[String]], isSorted: Boolean): Table =
    keyBy(maybeKeys.getOrElse(FastIndexedSeq()), isSorted)

  def unkey(): Table = keyBy(FastIndexedSeq())

  def mapRows(expr: String): Table =
    mapRows(IRParser.parse_value_ir(expr, IRParserEnvironment(typ.refMap)))

  def mapRows(newRow: IR): Table =
    new Table(hc, TableMapRows(tir, newRow))

  def export(path: String, typesFile: String = null, header: Boolean = true, exportType: Int = ExportType.CONCATENATED, delimiter: String = "\t") {
    ExecuteContext.scoped { ctx =>
      ir.Interpret[Unit](ctx, ir.TableWrite(tir, ir.TableTextWriter(path, typesFile, header, exportType, delimiter)))
    }
  }

  def distinctByKey(): Table = {
    new Table(hc, ir.TableDistinct(tir))
  }

  def explode(column: String): Table = new Table(hc, TableExplode(tir, Array(column)))

  def explode(columnNames: Array[String]): Table = {
    columnNames.foldLeft(this)((kt, name) => kt.explode(name))
  }

  def collect(): Array[Row] = rdd.collect()

  def write(path: String, overwrite: Boolean = false, stageLocally: Boolean = false, codecSpecJSONStr: String = null) {
    ExecuteContext.scoped { ctx =>
      ir.Interpret[Unit](ctx, ir.TableWrite(tir, TableNativeWriter(path, overwrite, stageLocally, codecSpecJSONStr)))
    }
  }

  def copy2(rvd: RVD = rvd,
    signature: TStruct = signature,
    key: IndexedSeq[String] = key,
    globalSignature: TStruct = globalSignature,
    globals: BroadcastRow = null): Table = ExecuteContext.scoped { ctx =>
    new Table(hc,
      TableLiteral(TableValue(TableType(signature, key, globalSignature),
        Option(globals).getOrElse(BroadcastRow(ctx, this.globals, globalSignature)),
        rvd),
        ctx)
    )
  }
}