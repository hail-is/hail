package is.hail.expr.ir

import is.hail.asm4s.{HailClassLoader, theHailClassLoaderForSparkWorkers}
import is.hail.HailContext
import is.hail.annotations._
import is.hail.backend.spark.SparkTaskContext
import is.hail.backend.{BroadcastValue, ExecuteContext, HailTaskContext}
import is.hail.expr.TableAnnotationImpex
import is.hail.expr.ir.lowering.{RVDToTableStage, TableStage, TableStageToRVD}
import is.hail.io.fs.FS
import is.hail.types.physical.{PArray, PCanonicalArray, PCanonicalStruct, PStruct}
import is.hail.types.virtual.{Field, TArray, TStruct}
import is.hail.types.{MatrixType, TableType}
import is.hail.io.{BufferSpec, TypedCodecSpec, exportTypes}
import is.hail.rvd.{AbstractRVDSpec, RVD, RVDContext, RVDPartitioner, RVDType}
import is.hail.sparkextras.ContextRDD
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.storage.StorageLevel
import org.json4s.jackson.JsonMethods

object TableExecuteIntermediate {
  def apply(tv: TableValue): TableExecuteIntermediate = new TableValueIntermediate(tv)

  def apply(ts: TableStage): TableExecuteIntermediate = new TableStageIntermediate(ts)
}

sealed trait TableExecuteIntermediate {
  def asTableStage(ctx: ExecuteContext): TableStage

  def asTableValue(ctx: ExecuteContext): TableValue

  def partitioner: RVDPartitioner
}

case class TableValueIntermediate(tv: TableValue) extends TableExecuteIntermediate {
  def asTableStage(ctx: ExecuteContext): TableStage = {
    RVDToTableStage(tv.rvd, tv.globals.toEncodedLiteral(ctx.theHailClassLoader))
  }

  def asTableValue(ctx: ExecuteContext): TableValue = tv

  def partitioner: RVDPartitioner = tv.rvd.partitioner
}

case class TableStageIntermediate(ts: TableStage) extends TableExecuteIntermediate {
  def asTableStage(ctx: ExecuteContext): TableStage = ts

  def asTableValue(ctx: ExecuteContext): TableValue = {
    val (globals, rvd) = TableStageToRVD(ctx, ts)
    TableValue(ctx, TableType(ts.rowType, ts.key, ts.globalType), globals, rvd)
  }

  def partitioner: RVDPartitioner = ts.partitioner
}


object TableValue {
  def apply(ctx: ExecuteContext, rowType: PStruct, key: IndexedSeq[String], rdd: ContextRDD[Long]): TableValue = {
    assert(rowType.required)
    val tt = TableType(rowType.virtualType, key, TStruct.empty)
    TableValue(ctx,
      tt,
      BroadcastRow.empty(ctx),
      RVD.coerce(ctx, RVDType(rowType, key), rdd))
  }

  def apply(ctx: ExecuteContext, rowType: TStruct, key: IndexedSeq[String], rdd: RDD[Row], rowPType: Option[PStruct] = None): TableValue = {
    val canonicalRowType = rowPType.getOrElse(PCanonicalStruct.canonical(rowType).setRequired(true).asInstanceOf[PStruct])
    assert(canonicalRowType.required)
    val tt = TableType(rowType, key, TStruct.empty)
    TableValue(ctx,
      tt,
      BroadcastRow.empty(ctx),
      RVD.coerce(ctx,
        RVDType(canonicalRowType, key),
        ContextRDD.weaken(rdd).toRegionValues(canonicalRowType)))
  }
}

case class TableValue(ctx: ExecuteContext, typ: TableType, globals: BroadcastRow, rvd: RVD) {
  if (typ.rowType != rvd.rowType)
    throw new RuntimeException(s"row mismatch:\n  typ: ${ typ.rowType.parsableString() }\n  rvd: ${ rvd.rowType.parsableString() }")
  if (!rvd.typ.key.startsWith(typ.key))
    throw new RuntimeException(s"key mismatch:\n  typ: ${ typ.key }\n  rvd: ${ rvd.typ.key }")
  if (typ.globalType != globals.t.virtualType)
    throw new RuntimeException(s"globals mismatch:\n  typ: ${ typ.globalType.parsableString() }\n  val: ${ globals.t.virtualType.parsableString() }")
  if (!globals.t.required)
    throw new RuntimeException(s"globals not required; ${ globals.t }")

  def rdd: RDD[Row] =
    rvd.toRows

  def persist(ctx: ExecuteContext, level: StorageLevel) =
    TableValue(ctx, typ, globals, rvd.persist(ctx, level))

  def filterWithPartitionOp[P](theHailClassLoader: HailClassLoader, fs: BroadcastValue[FS], partitionOp: (HailClassLoader, FS, HailTaskContext, Region) => P)(pred: (P, RVDContext, Long, Long) => Boolean): TableValue = {
    val localGlobals = globals.broadcast(theHailClassLoader)
    copy(rvd = rvd.filterWithContext[(P, Long)](
      { (partitionIdx, ctx) =>
        val globalRegion = ctx.partitionRegion
        (
          partitionOp(theHailClassLoaderForSparkWorkers, fs.value, SparkTaskContext.get(), globalRegion),
          localGlobals.value.readRegionValue(globalRegion, theHailClassLoaderForSparkWorkers)
        )
      }, { case ((p, glob), ctx, ptr) => pred(p, ctx, ptr, glob) }))
  }

  def filter(theHailClassLoader: HailClassLoader, fs: BroadcastValue[FS], p: (RVDContext, Long, Long) => Boolean): TableValue = {
    filterWithPartitionOp(theHailClassLoader, fs, (_, _, _, _) => ())((_, ctx, ptr, glob) => p(ctx, ptr, glob))
  }

  def export(ctx: ExecuteContext, path: String, typesFile: String = null, header: Boolean = true, exportType: String = ExportType.CONCATENATED, delimiter: String = "\t") {
    val fs = ctx.fs
    fs.delete(path, recursive = true)

    val fields = typ.rowType.fields

    Option(typesFile).foreach { file =>
      exportTypes(file, fs, fields.map(f => (f.name, f.typ)).toArray)
    }

    val localSignature = rvd.rowPType
    val localTypes = fields.map(_.typ)

    val localDelim = delimiter
    rvd.mapPartitions { (ctx, it) =>
      val sb = new StringBuilder()

      it.map { ptr =>
        val ur = new UnsafeRow(localSignature, ctx.r, ptr)
        sb.clear()
        localTypes.indices.foreachBetween { i =>
          sb.append(TableAnnotationImpex.exportAnnotation(ur.get(i), localTypes(i)))
        }(sb.append(localDelim))

        sb.result()
      }
    }.writeTable(ctx, path, Some(fields.map(_.name).mkString(localDelim)).filter(_ => header), exportType = exportType)
  }

  def toDF(): DataFrame = {
    HailContext.sparkBackend("toDF").sparkSession.createDataFrame(
      rvd.toRows,
      typ.rowType.schema.asInstanceOf[StructType])
  }

  def rename(globalMap: Map[String, String], rowMap: Map[String, String]): TableValue = {
    TableValue(ctx,
      typ.copy(
        rowType = typ.rowType.rename(rowMap),
        globalType = typ.globalType.rename(globalMap),
        key = typ.key.map(k => rowMap.getOrElse(k, k))),
      globals.copy(t = globals.t.rename(globalMap)), rvd = rvd.cast(rvd.rowPType.rename(rowMap)))
  }

  def toMatrixValue(colKey: IndexedSeq[String],
    colsFieldName: String = LowerMatrixIR.colsFieldName,
    entriesFieldName: String = LowerMatrixIR.entriesFieldName): MatrixValue = {

    val (colType, colsFieldIdx) = typ.globalType.field(colsFieldName) match {
      case Field(_, TArray(t@TStruct(_)), idx) => (t, idx)
      case Field(_, t, _) => fatal(s"expected cols field to be an array of structs, found $t")
    }

    val mType: MatrixType = MatrixType(
      typ.globalType.deleteKey(colsFieldName, colsFieldIdx),
      colKey,
      colType,
      typ.key,
      typ.rowType.deleteKey(entriesFieldName),
      typ.rowType.field(MatrixType.entriesIdentifier).typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct])

    val globalsT = globals.t
    val colsT = globalsT.field(colsFieldName).typ.asInstanceOf[PArray]

    val globals2 =
      if (colsT.required && colsT.elementType.required)
        globals
      else
        globals.cast(
          globalsT.insertFields(FastIndexedSeq(
            colsFieldName -> PCanonicalArray(colsT.elementType.setRequired(true), true))))

    val newTV = TableValue(ctx, typ, globals2, rvd)

    MatrixValue(mType, newTV.rename(
      Map(colsFieldName -> LowerMatrixIR.colsFieldName),
      Map(entriesFieldName -> LowerMatrixIR.entriesFieldName)))
  }
}
