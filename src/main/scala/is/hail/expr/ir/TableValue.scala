package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations.{BroadcastRow, RegionValue, RegionValueBuilder, UnsafeRow}
import is.hail.expr.TableAnnotationImpex
import is.hail.expr.types.TableType
import is.hail.io.{CodecSpec, exportTypes}
import is.hail.rvd.{OrderedRVD, OrderedRVDType, RVD, RVDSpec, UnpartitionedRVD}
import is.hail.table.TableSpec
import is.hail.utils._
import is.hail.variant.{FileFormat, PartitionCountsComponentSpec, RVDComponentSpec, ReferenceGenome}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.json4s.jackson.JsonMethods

case class TableValue(typ: TableType, globals: BroadcastRow, rvd: RVD) {
  require(typ.rowType == rvd.rowType)
  require(typ.key.isDefined == rvd.isInstanceOf[OrderedRVD])
  if (typ.key.exists(k => !rvd.asInstanceOf[OrderedRVD].typ.key.startsWith(k)))
    println(s"type key: ${typ.key.get}, rvd key: ${rvd.asInstanceOf[OrderedRVD].typ.key}")
  require(typ.key.forall(k => rvd.asInstanceOf[OrderedRVD].typ.key.startsWith(k)))

  def rdd: RDD[Row] =
    rvd.toRows

  def keyedRDD(): RDD[(Row, Row)] = {
    require(typ.key.isDefined)
    val fieldIndices = typ.rowType.fields.map(f => f.name -> f.index).toMap
    val keyIndices = typ.key.get.map(fieldIndices)
    val keyIndexSet = keyIndices.toSet
    val valueIndices = typ.rowType.fields.filter(f => !keyIndexSet.contains(f.index)).map(_.index)
    rdd.map { r => (Row.fromSeq(keyIndices.map(r.get)), Row.fromSeq(valueIndices.map(r.get))) }
  }

  def enforceOrderingRVD: RVD = (rvd, typ.key) match {
    case (orvd: OrderedRVD, Some(key)) =>
      assert(orvd.typ.key.startsWith(key))
      orvd
    case (orvd: OrderedRVD, None) =>
      orvd.toUnpartitionedRVD
    case (urvd: UnpartitionedRVD, Some(key)) =>
      val orvdType = OrderedRVDType(key, typ.rowType)
      OrderedRVD.coerce(orvdType, rvd)
    case (urvd: UnpartitionedRVD, None) =>
      urvd
  }

  def filterWithPartitionOp[P](partitionOp: Int => P)(pred: (P, RegionValue, RegionValue) => Boolean): TableValue = {
    val globalType = typ.globalType
    val localGlobals = globals.broadcast
    copy(rvd = rvd.filterWithContext[(P, RegionValue)](
      { (partitionIdx, ctx) =>
        val globalRegion = ctx.freshRegion
        val rvb = new RegionValueBuilder()
        rvb.set(globalRegion)
        rvb.start(globalType)
        rvb.addAnnotation(globalType, localGlobals.value)
        (partitionOp(partitionIdx), RegionValue(globalRegion, rvb.end()))
      }, { case ((p, glob), rv) => pred(p, rv, glob) }))
  }

  def filter(p: (RegionValue, RegionValue) => Boolean): TableValue = {
    filterWithPartitionOp(_ => ())((_, rv1, rv2) => p(rv1, rv2))
  }

  def write(path: String, overwrite: Boolean, stageLocally: Boolean, codecSpecJSONStr: String) {
    val hc = HailContext.get

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
    RVD.writeLocalUnpartitioned(hc, globalsPath, typ.globalType, codecSpec, Array(globals.value))

    val partitionCounts = enforceOrderingRVD.write(path + "/rows", stageLocally, codecSpec)

    val referencesPath = path + "/references"
    hc.hadoopConf.mkDir(referencesPath)
    ReferenceGenome.exportReferences(hc, referencesPath, typ.rowType)
    ReferenceGenome.exportReferences(hc, referencesPath, typ.globalType)

    val spec = TableSpec(
      FileFormat.version.rep,
      hc.version,
      "references",
      typ,
      Map("globals" -> RVDComponentSpec("globals"),
        "rows" -> RVDComponentSpec("rows"),
        "partition_counts" -> PartitionCountsComponentSpec(partitionCounts)))
    spec.write(hc, path)

    hc.hadoopConf.writeTextFile(path + "/_SUCCESS")(out => ())
  }

  def export(path: String, typesFile: String = null, header: Boolean = true, exportType: Int = ExportType.CONCATENATED) {
    val hc = HailContext.get
    hc.hadoopConf.delete(path, recursive = true)

    val fields = typ.rowType.fields

    Option(typesFile).foreach { file =>
      exportTypes(file, hc.hadoopConf, fields.map(f => (f.name, f.typ)).toArray)
    }

    val localSignature = typ.rowType
    val localTypes = fields.map(_.typ)

    rvd.mapPartitions { it =>
      val sb = new StringBuilder()

      it.map { rv =>
        val ur = new UnsafeRow(localSignature, rv)
        sb.clear()
        localTypes.indices.foreachBetween { i =>
          sb.append(TableAnnotationImpex.exportAnnotation(ur.get(i), localTypes(i)))
        }(sb += '\t')

        sb.result()
      }
    }.writeTable(path, hc.tmpDir, Some(fields.map(_.name).mkString("\t")).filter(_ => header), exportType = exportType)
  }
}
