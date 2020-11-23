package is.hail.types.physical.stypes

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.{PCode, PType}


trait SType {
  def pType: PType

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]]

  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering
}
