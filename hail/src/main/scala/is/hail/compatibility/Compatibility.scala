package is.hail.compatibility

import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.types._
import is.hail.expr.types.physical.PStruct
import is.hail.io.CodecSpec
import is.hail.rvd.{RVDSpec, RVDPartitioner, RVDType}
import is.hail.table.TableSpec
import is.hail.utils.{FastIndexedSeq, Interval, SemanticVersion}
import is.hail.variant._
import org.json4s.JsonAST.{JInt, JNothing, JString}
import org.json4s.{Formats, JValue, ShortTypeHints}

trait JsonIntermediate[T] {
  def spec: T
}

case class MatrixTableSpec_1_0(
  file_version: Int,
  hail_version: String,
  references_rel_path: String,
  matrix_type: MatrixType,
  components: Map[String, ComponentSpec]) extends MatrixTableSpec with JsonIntermediate[RelationalSpec] {

  def spec: RelationalSpec = this
}

case class TableSpec_1_0(
  file_version: Int,
  hail_version: String,
  references_rel_path: String,
  table_type: TableType,
  components: Map[String, ComponentSpec]) extends TableSpec with JsonIntermediate[RelationalSpec] {

  def spec: RelationalSpec = this
}

case class OrderedRVDSpec_1_0(
  rvdType: RVDType,
  codecSpec: CodecSpec,
  partFiles: Array[String],
  jRangeBounds: JValue
) extends JsonIntermediate[RVDSpec] {
  def spec: RVDSpec = {
    val rangeBoundsType = TArray(TInterval(rvdType.kType.virtualType))
    val partitioner = new RVDPartitioner(rvdType.kType.virtualType,
      JSONAnnotationImpex.importAnnotation(jRangeBounds, rangeBoundsType, padNulls = false).asInstanceOf[IndexedSeq[Interval]])
    RVDSpec(
      rvdType.rowType,
      rvdType.key,
      codecSpec,
      partFiles,
      partitioner)
  }
}

case class RVDSpec_1_1(
  file_version: Int,
  rowType: TStruct,
  key: IndexedSeq[String],
  codecSpec: CodecSpec,
  partFiles: Array[String],
  jRangeBounds: JValue
) extends RVDSpec with JsonIntermediate[RVDSpec] {
  require({
    val v = SemanticVersion(file_version)
    v.major == 1 && v.minor == 1
  })

  override def encodedType: PStruct = rowType.physicalType

  def partitioner: RVDPartitioner = {
    val kType = rowType.typeAfterSelect(key.map(encodedType.fieldIdx))
    val rangeBoundsType = TArray(TInterval(kType))
    new RVDPartitioner(kType,
      JSONAnnotationImpex.importAnnotation(jRangeBounds, rangeBoundsType, padNulls = false).asInstanceOf[IndexedSeq[Interval]])
  }

  def spec: RVDSpec = this
}


case class UnpartitionedRVDSpec_1_0(
  rowType: TStruct,
  codecSpec: CodecSpec,
  partFiles: Array[String]
) extends JsonIntermediate[RVDSpec] {
  def spec: RVDSpec = RVDSpec(rowType.physicalType, FastIndexedSeq(), codecSpec, partFiles, RVDPartitioner.unkeyed(partFiles.length))
}

object Compatibility {
  def extractRel(fileVersion: SemanticVersion, jv: JValue): RelationalSpec = {
    implicit val formats: Formats = RelationalSpec.formats + ShortTypeHints(List(classOf[MatrixTableSpec_1_0], classOf[TableSpec_1_0]))
    fileVersion match {
      case SemanticVersion(1, 1, 0) => jv.extract[JsonIntermediate[RelationalSpec]].spec
      case SemanticVersion(1, 0, _) =>
        jv.transformField { case ("name", JString("MatrixTableSpec")) => ("name", JString("MatrixTableSpec_1_0")) }
          .transformField { case ("name", JString("TableSpec")) => ("name", JString("TableSpec_1_0")) }
          .extract[JsonIntermediate[RelationalSpec]].spec
    }
  }

  def extractRVD(jv: JValue): RVDSpec = {
    val fileVersion: Option[SemanticVersion] = jv \ "file_version" match {
      case JInt(rep) => Some(SemanticVersion(rep.toInt))
      case JNothing => None
    }
    fileVersion.foreach(v => assert(FileFormat.version.supports(v)))

    implicit val formats: Formats = RVDSpec.formats + ShortTypeHints(List(
      classOf[UnpartitionedRVDSpec_1_0],
      classOf[OrderedRVDSpec_1_0],
      classOf[RVDSpec_1_1]))

    fileVersion match {
      case Some(SemanticVersion(1, 1, _)) => jv.extract[JsonIntermediate[RVDSpec]].spec
      case Some(SemanticVersion(1, 0, _)) | None =>
        jv.transformField { case ("orvdType", value) => ("rvdType", value) }
          .transformField { case ("name", JString("UnpartitionedRVDSpec")) => ("name", JString("UnpartitionedRVDSpec_1_0")) }
          .transformField { case ("name", JString("OrderedRVDSpec")) => ("name", JString("OrderedRVDSpec_1_0")) }
          .extract[JsonIntermediate[RVDSpec]].spec
    }
  }
}
