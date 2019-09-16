package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.TableAnnotationImpex
import is.hail.expr.types.physical.PStruct
import is.hail.expr.types.{MatrixType, TableType}
import is.hail.expr.types.virtual.{Field, TArray, TStruct}
import is.hail.io.{CodecSpec, exportTypes}
import is.hail.rvd.{AbstractRVDSpec, RVD, RVDContext, RVDType}
import is.hail.sparkextras.ContextRDD
import is.hail.table.TableSpec
import is.hail.utils._
import is.hail.variant.{FileFormat, PartitionCountsComponentSpec, RVDComponentSpec, ReferenceGenome}
import is.hail.io.fs.FS
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.storage.StorageLevel
import org.json4s.jackson.JsonMethods

object TableValue {
  def apply(ctx: ExecuteContext, rowType: PStruct, key: IndexedSeq[String], rdd: ContextRDD[RVDContext, RegionValue]): TableValue = {
    val tt = TableType(rowType.virtualType, key, TStruct.empty())
    TableValue(tt,
      BroadcastRow.empty(ctx),
      RVD.coerce(RVDType(rowType, key), rdd))
  }

  def apply(ctx: ExecuteContext, rowType: TStruct, key: IndexedSeq[String], rdd: ContextRDD[RVDContext, RegionValue]): TableValue = {
    val tt = TableType(rowType, key, TStruct.empty())
    TableValue(tt,
        BroadcastRow.empty(ctx),
        RVD.coerce(tt.canonicalRVDType, rdd))
  }

  def apply(ctx: ExecuteContext, rowType:  TStruct, key: IndexedSeq[String], rdd: RDD[Row]): TableValue = {
    val canonicalRowType = PStruct.canonical(rowType)
    val tt = TableType(rowType, key, TStruct.empty())
    TableValue(tt,
      BroadcastRow.empty(ctx),
      RVD.coerce(RVDType(canonicalRowType, key), ContextRDD.weaken[RVDContext](rdd).toRegionValues(canonicalRowType)))
  }
}

case class TableValue(typ: TableType, globals: BroadcastRow, rvd: RVD) {
  require(typ.rowType == rvd.rowType, s"mismatch:\n  typ: ${ typ.rowType }\n  rvd: ${ rvd.rowType }")
  require(rvd.typ.key.startsWith(typ.key))
  require(typ.globalType == globals.t.virtualType)

  def rdd: RDD[Row] =
    rvd.toRows

  def filterWithPartitionOp[P](partitionOp: (Int, Region) => P)(pred: (P, RegionValue, RegionValue) => Boolean): TableValue = {
    val localGlobals = globals.broadcast
    copy(rvd = rvd.filterWithContext[(P, RegionValue)](
      { (partitionIdx, ctx) =>
        val globalRegion = ctx.freshRegion
        (partitionOp(partitionIdx, globalRegion), RegionValue(globalRegion, localGlobals.value.readRegionValue(globalRegion)))
      }, { case ((p, glob), rv) => pred(p, rv, glob) }))
  }

  def filter(p: (RegionValue, RegionValue) => Boolean): TableValue = {
    filterWithPartitionOp((_, _) => ())((_, rv1, rv2) => p(rv1, rv2))
  }

  def write(path: String, overwrite: Boolean, stageLocally: Boolean, codecSpecJSONStr: String) {
    assert(typ.isCanonical)
    val hc = HailContext.get
    val fs = hc.sFS

    val codecSpec =
      if (codecSpecJSONStr != null) {
        implicit val formats = AbstractRVDSpec.formats
        val codecSpecJSON = JsonMethods.parse(codecSpecJSONStr)
        codecSpecJSON.extract[CodecSpec]
      } else
        CodecSpec.default

    if (overwrite)
      fs.delete(path, recursive = true)
    else if (fs.exists(path))
      fatal(s"file already exists: $path")

    fs.mkDir(path)

    val globalsPath = path + "/globals"
    fs.mkDir(globalsPath)
    AbstractRVDSpec.writeSingle(fs, globalsPath, globals.t, codecSpec, Array(globals.javaValue))

    val partitionCounts = rvd.write(path + "/rows", "../index", stageLocally, codecSpec)

    val referencesPath = path + "/references"
    fs.mkDir(referencesPath)
    ReferenceGenome.exportReferences(fs, referencesPath, typ.rowType)
    ReferenceGenome.exportReferences(fs, referencesPath, typ.globalType)

    val spec = TableSpec(
      FileFormat.version.rep,
      is.hail.HAIL_PRETTY_VERSION,
      "references",
      typ,
      Map("globals" -> RVDComponentSpec("globals"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    spec.write(fs, path)

    writeNativeFileReadMe(path)

    fs.writeTextFile(path + "/_SUCCESS")(out => ())

    val nRows = partitionCounts.sum
    info(s"wrote table with $nRows ${ plural(nRows, "row") } " +
      s"in ${ partitionCounts.length } ${ plural(partitionCounts.length, "partition") } " +
      s"to $path")

  }

  def export(path: String, typesFile: String = null, header: Boolean = true, exportType: Int = ExportType.CONCATENATED, delimiter: String = "\t") {
    val hc = HailContext.get
    hc.sFS.delete(path, recursive = true)

    val fields = typ.rowType.fields

    Option(typesFile).foreach { file =>
      exportTypes(file, hc.sFS, fields.map(f => (f.name, f.typ)).toArray)
    }

    val localSignature = rvd.rowPType
    val localTypes = fields.map(_.typ)

    val localDelim = delimiter
    rvd.mapPartitions { it =>
      val sb = new StringBuilder()

      it.map { rv =>
        val ur = new UnsafeRow(localSignature, rv)
        sb.clear()
        localTypes.indices.foreachBetween { i =>
          sb.append(TableAnnotationImpex.exportAnnotation(ur.get(i), localTypes(i)))
        }(sb.append(localDelim))

        sb.result()
      }
    }.writeTable(hc.sFS, path, hc.tmpDir, Some(fields.map(_.name).mkString(localDelim)).filter(_ => header), exportType = exportType)
  }

  def toDF(): DataFrame = {
    HailContext.get.sparkSession.createDataFrame(
      rvd.toRows,
      typ.rowType.schema.asInstanceOf[StructType])
  }

  def rename(globalMap: Map[String, String], rowMap: Map[String, String]): TableValue = {
    TableValue(typ, globals.copy(t = globals.t.rename(globalMap)), rvd = rvd.cast(rvd.rowPType.rename(rowMap)))
  }

  def toMatrixValue(colKey: IndexedSeq[String],
    colsFieldName: String = LowerMatrixIR.colsFieldName,
    entriesFieldName: String = LowerMatrixIR.entriesFieldName): MatrixValue = {

    val (colType, colsFieldIdx) = typ.globalType.field(colsFieldName) match {
      case Field(_, TArray(t@TStruct(_, _), _), idx) => (t, idx)
      case Field(_, t, _) => fatal(s"expected cols field to be an array of structs, found $t")
    }

    val mType: MatrixType = MatrixType(
      typ.globalType.deleteKey(colsFieldName, colsFieldIdx),
      colKey,
      colType,
      typ.key,
      typ.rowType.deleteKey(entriesFieldName),
      typ.rowType.field(MatrixType.entriesIdentifier).typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct])

    MatrixValue(mType, rename(
      Map(colsFieldName -> LowerMatrixIR.colsFieldName),
      Map(entriesFieldName -> LowerMatrixIR.entriesFieldName)))
  }
}