package is.hail.types.physical.mtypes

import is.hail.expr.ir.EmitCodeBuilder

trait MStruct extends MType {
  def fields: IndexedSeq[MField]
  def getField(cb: EmitCodeBuilder, idx: Int, mv: MValue): IEmitMCode
}
