package is.hail.compatibility

import is.hail.expr.types.virtual.TStruct
import is.hail.io.{CodecSpec, CodecSpec2, PackCodecSpec, PackCodecSpec2}
import is.hail.rvd.{AbstractRVDSpec, RVDPartitioner}
import is.hail.utils.FastIndexedSeq


case class UnpartitionedRVDSpec(
  rowType: TStruct,
  codecSpec: CodecSpec,
  partFiles: Array[String]
) extends AbstractRVDSpec {
  def partitioner: RVDPartitioner = RVDPartitioner.unkeyed(partFiles.length)

  def key: IndexedSeq[String] = FastIndexedSeq()

  def encodedType: TStruct = rowType

  def codecSpec2: CodecSpec2 = PackCodecSpec2(encodedType.physicalType, codecSpec.asInstanceOf[PackCodecSpec].child)
}