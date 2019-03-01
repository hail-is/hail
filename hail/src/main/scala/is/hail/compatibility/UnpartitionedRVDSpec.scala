package is.hail.compatibility

import is.hail.{HailContext, cxx}
import is.hail.expr.types.physical.PStruct
import is.hail.expr.types.virtual.TStruct
import is.hail.io.CodecSpec
import is.hail.rvd.{AbstractRVDSpec, RVDPartitioner, RVDType}
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