package is.hail.compatibility

import is.hail.expr.types._
import is.hail.expr.types.physical.PStruct
import is.hail.io.CodecSpec
import is.hail.rvd.{AbstractRVDSpec, RVDPartitioner}
import is.hail.utils.FastIndexedSeq
import is.hail.variant._
import org.json4s.{Formats, JValue, ShortTypeHints}


case class UnpartitionedRVDSpec(
  rowType: TStruct,
  codecSpec: CodecSpec,
  partFiles: Array[String]
) extends AbstractRVDSpec {
  def partitioner: RVDPartitioner = RVDPartitioner.unkeyed(partFiles.length)

  def key: IndexedSeq[String] = FastIndexedSeq()

  def encodedType: PStruct = rowType.physicalType
}

object Compatibility {
  def extractRel(jv: JValue): RelationalSpec = {
    implicit val formats: Formats = RelationalSpec.formats

    jv.extract[RelationalSpec]
  }

  def extractRVD(jv: JValue): AbstractRVDSpec = {
    implicit val formats: Formats = AbstractRVDSpec.formats + ShortTypeHints(List(
      classOf[UnpartitionedRVDSpec]))

    jv.transformField { case ("orvdType", value) => ("rvdType", value) }
      .extract[AbstractRVDSpec]
  }
}
