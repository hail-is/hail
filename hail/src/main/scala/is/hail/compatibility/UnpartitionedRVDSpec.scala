package is.hail.compatibility

import is.hail.expr.ir.Sym
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

  def key: IndexedSeq[Sym] = FastIndexedSeq()

  def encodedType: PStruct = rowType.physicalType

  def cxxEmitRead(hc: HailContext, path: String, requestedType: TStruct, tub: cxx.TranslationUnitBuilder): cxx.RVDEmitTriplet = {
    cxx.RVDEmitTriplet.read(path, encodedType, codecSpec, partFiles, RVDType(requestedType.physicalType, FastIndexedSeq()), null, tub)
  }
}