package is.hail.compatibility

import is.hail.expr.types._
import is.hail.expr.types.physical.PStruct
import is.hail.io.CodecSpec
import is.hail.rvd.{AbstractRVDSpec, RVDPartitioner}
import is.hail.utils.FastIndexedSeq


case class UnpartitionedRVDSpec(
  rowType: TStruct,
  codecSpec: CodecSpec,
  partFiles: Array[String]
) extends AbstractRVDSpec {
  def partitioner: RVDPartitioner = RVDPartitioner.unkeyed(partFiles.length)

  def key: IndexedSeq[String] = FastIndexedSeq()

  def encodedType: PStruct = rowType.physicalType
}