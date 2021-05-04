package is.hail.types.physical.stypes

import is.hail.annotations.Region
import is.hail.asm4s.{Code, Settable, TypeInfo, Value}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, EmitParamType, InferPType, PCodeEmitParamType, PCodeParamType, SortOrder}
import is.hail.types.TypeWithRequiredness
import is.hail.types.physical.{PCode, PType}
import is.hail.types.virtual.Type


object SType {
  def chooseCompatibleType(req: TypeWithRequiredness, stypes: SType*): SType = {
    InferPType.getCompatiblePType(stypes.map(_.pType), req).sType
  }
}

trait SType {
  def virtualType: Type = pType.virtualType

  def pType: PType

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]]

  def fromSettables(settables: IndexedSeq[Settable[_]]): SSettable

  def fromCodes(codes: IndexedSeq[Code[_]]): SCode

  def canonicalPType(): PType

  def paramType: PCodeParamType = PCodeParamType(pType)

  def asIdent: String = pType.asIdent

  def equalsExceptTopLevelRequiredness(that: SType): Boolean = pType.equalModuloRequired(that.pType)
}

case class EmitType(st: SType, required: Boolean) {
  def virtualType: Type = st.virtualType
  def paramType: EmitParamType = PCodeEmitParamType(st.pType.setRequired(required))
}