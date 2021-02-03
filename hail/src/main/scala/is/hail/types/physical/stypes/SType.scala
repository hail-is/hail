package is.hail.types.physical.stypes

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, Settable, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.TypeWithRequiredness
import is.hail.types.physical.{PCode, PType}
import is.hail.types.virtual.Type

trait SType {
  def virtualType: Type = pType.virtualType

  def pType: PType

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]]

  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering

  def fromSettables(settables: IndexedSeq[Settable[_]]): SSettable

  def fromCodes(codes: IndexedSeq[Code[_]]): SCode

  def canonicalPType(): PType
}
