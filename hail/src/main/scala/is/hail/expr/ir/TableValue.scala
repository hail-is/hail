package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.TableAnnotationImpex
import is.hail.expr.types.{MatrixType, TableType}
import is.hail.expr.types.virtual.{Field, TArray, TStruct}
import is.hail.io.{CodecSpec, exportTypes}
import is.hail.rvd.{AbstractRVDSpec, RVD, RVDContext}
import is.hail.sparkextras.ContextRDD
import is.hail.table.TableSpec
import is.hail.utils._
import is.hail.variant.{FileFormat, PartitionCountsComponentSpec, RVDComponentSpec, ReferenceGenome}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.storage.StorageLevel
import org.json4s.jackson.JsonMethods

object TableValue {
  def apply(rowType: TStruct, key: IndexedSeq[String], rdd: ContextRDD[RVDContext, RegionValue]): TableValue = {
    Interpret(
      TableKeyBy(TableLiteral(TableValue(TableType(rowType, FastIndexedSeq(), TStruct.empty()),
        BroadcastRow.empty(),
        RVD.unkeyed(rowType.physicalType, rdd))),
        key))
  }

  def apply(rowType:  TStruct, key: IndexedSeq[String], rdd: RDD[Row]): TableValue =
    apply(rowType, key, ContextRDD.weaken[RVDContext](rdd).toRegionValues(rowType))

  def apply(typ: TableType, globals: BroadcastRow, rdd: RDD[Row]): TableValue =
    Interpret(
      TableKeyBy(TableLiteral(TableValue(typ.copy(key = FastIndexedSeq()), globals,
      RVD.unkeyed(typ.rowType.physicalType, ContextRDD.weaken[RVDContext](rdd).toRegionValues(typ.rowType)))),
        typ.key))
}

case class TableValue(typ: TableType, globals: BroadcastRow, rvd: RVD) {
  require(typ.rowType == rvd.rowType)
  require(rvd.typ.key.startsWith(typ.key))

  def rdd: RDD[Row] =
    rvd.toRows

  def keyedRDD(): RDD[(Row, Row)] = {
    val fieldIndices = typ.rowType.fields.map(f => f.name -> f.index).toMap
    val keyIndices = typ.key.map(fieldIndices)
    val keyIndexSet = keyIndices.toSet
    val valueIndices = typ.rowType.fields.filter(f => !keyIndexSet.contains(f.index)).map(_.index)
    rdd.map { r => (Row.fromSeq(keyIndices.map(r.get)), Row.fromSeq(valueIndices.map(r.get))) }
  }

  def filterWithPartitionOp[P](partitionOp: Int => P)(pred: (P, RegionValue, RegionValue) => Boolean): TableValue = {
    val globalType = typ.globalType
    val localGlobals = globals.broadcast
    copy(rvd = rvd.filterWithContext[(P, RegionValue)](
      { (partitionIdx, ctx) =>
        val globalRegion = ctx.freshRegion
        val rvb = new RegionValueBuilder()
        rvb.set(globalRegion)
        rvb.start(globalType.physicalType)
        rvb.addAnnotation(globalType, localGlobals.value)
        (partitionOp(partitionIdx), RegionValue(globalRegion, rvb.end()))
      }, { case ((p, glob), rv) => pred(p, rv, glob) }))
  }

  def filter(p: (RegionValue, RegionValue) => Boolean): TableValue = {
    filterWithPartitionOp(_ => ())((_, rv1, rv2) => p(rv1, rv2))
  }

  def write(path: String, overwrite: Boolean, stageLocally: Boolean, codecSpecJSONStr: String) {
    val hc = HailContext.get
    val hadoopConf = hc.hadoopConf

    val codecSpec =
      if (codecSpecJSONStr != null) {
        implicit val formats = AbstractRVDSpec.formats
        val codecSpecJSON = JsonMethods.parse(codecSpecJSONStr)
        codecSpecJSON.extract[CodecSpec]
      } else
        CodecSpec.default

    if (overwrite)
      hadoopConf.delete(path, recursive = true)
    else if (hadoopConf.exists(path))
      fatal(s"file already exists: $path")

    hadoopConf.mkDir(path)

    val globalsPath = path + "/globals"
    hadoopConf.mkDir(globalsPath)
    AbstractRVDSpec.writeSingle(hadoopConf, globalsPath, typ.globalType.physicalType, codecSpec, Array(globals.value))

    val partitionCounts = rvd.write(path + "/rows", stageLocally, codecSpec)

    val referencesPath = path + "/references"
    hadoopConf.mkDir(referencesPath)
    ReferenceGenome.exportReferences(hadoopConf, referencesPath, typ.rowType)
    ReferenceGenome.exportReferences(hadoopConf, referencesPath, typ.globalType)

    val spec = TableSpec(
      FileFormat.version.rep,
      is.hail.HAIL_PRETTY_VERSION,
      "references",
      typ,
      Map("globals" -> RVDComponentSpec("globals"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    spec.write(hadoopConf, path)

    writeNativeFileReadMe(path)

    hadoopConf.writeTextFile(path + "/_SUCCESS")(out => ())

    val nRows = partitionCounts.sum
    info(s"wrote table with $nRows ${ plural(nRows, "row") } " +
      s"in ${ partitionCounts.length } ${ plural(partitionCounts.length, "partition") } " +
      s"to $path")

  }

  def export(path: String, typesFile: String = null, header: Boolean = true, exportType: Int = ExportType.CONCATENATED, delimiter: String = "\t") {
    val hc = HailContext.get
    hc.hadoopConf.delete(path, recursive = true)

    val fields = typ.rowType.fields

    Option(typesFile).foreach { file =>
      exportTypes(file, hc.hadoopConf, fields.map(f => (f.name, f.typ)).toArray)
    }

    val localSignature = typ.rowType.physicalType
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
    }.writeTable(path, hc.tmpDir, Some(fields.map(_.name).mkString(localDelim)).filter(_ => header), exportType = exportType)
  }

  def persist(storageLevel: StorageLevel): TableValue = copy(rvd = rvd.persist(storageLevel))

  def unpersist(): TableValue = copy(rvd = rvd.unpersist())

  def toDF(): DataFrame = {
    HailContext.get.sparkSession.createDataFrame(
      rvd.toRows,
      typ.rowType.schema.asInstanceOf[StructType])
  }

  def toMatrixValue(colsFieldName: String, entriesFieldName: String, colKey: IndexedSeq[String]): MatrixValue = {

    val (colType, colsFieldIdx) = typ.globalType.field(colsFieldName) match {
      case Field(_, TArray(t@TStruct(_, _), _), idx) => (t, idx)
      case Field(_, t, _) => fatal(s"expected cols field to be an array of structs, found $t")
    }
    val m = Map(entriesFieldName -> MatrixType.entriesIdentifier)

    val mType: MatrixType = MatrixType(
      typ.globalType.deleteKey(colsFieldName, colsFieldIdx),
      colKey,
      colType,
      typ.key,
      typ.rowType.deleteKey(entriesFieldName),
      typ.rowType.field(MatrixType.entriesIdentifier).typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct])

    val colValues = globals.value.getAs[IndexedSeq[Annotation]](colsFieldIdx)
    val newGlobals = {
      val (pre, post) = globals.value.toSeq.splitAt(colsFieldIdx)
      Row.fromSeq(pre ++ post.tail)
    }

    val newRVD = rvd.cast(rvd.rowPType.rename(m))

    MatrixValue(
      mType,
      BroadcastRow(newGlobals, mType.globalType, HailContext.get.sc),
      BroadcastIndexedSeq(colValues, TArray(mType.colType), HailContext.get.sc),
      newRVD
    )
  }
}
