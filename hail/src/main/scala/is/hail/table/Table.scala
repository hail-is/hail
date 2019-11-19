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
import java.nio.charset.StandardCharsets
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.json4s.jackson.JsonMethods

import scala.collection.JavaConverters._
import scala.language.implicitConversions

object SortOrder {
  def deserialize(b: Byte): SortOrder =
    if (b == 0.toByte) Ascending
    else if (b == 1.toByte) Descending
    else throw new RuntimeException(s"invalid sort order: $b")
}

sealed abstract class SortOrder {
  def serialize: Byte
}

case object Ascending extends SortOrder {
  def serialize: Byte = 0.toByte
}

case object Descending extends SortOrder {
  def serialize: Byte = 1.toByte
}

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
    signature: TStruct,
    key: IndexedSeq[String]
  ): Table = apply(hc, rdd, signature, key, TStruct.empty(), Annotation.empty, isSorted = false)

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
}

class Table(val hc: HailContext, val tir: TableIR) {
  def typ: TableType = tir.typ

  lazy val (globals, rvd, rdd, lit) = {
    ExecuteContext.scoped { ctx =>
      val tv = Interpret(tir, ctx, optimize = true)
      (tv.globals.safeJavaValue, tv.rvd, tv.rdd, TableLiteral(tv, ctx))
    }
  }

  val TableType(signature, key, globalSignature) = tir.typ

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
