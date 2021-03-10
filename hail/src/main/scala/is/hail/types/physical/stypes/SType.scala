package is.hail.types.physical.stypes

import is.hail.annotations.Region
import is.hail.asm4s.{Code, Settable, TypeInfo, Value}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, EmitParamType, PCodeParamType, SortOrder}
import is.hail.types.TypeWithRequiredness
import is.hail.types.physical.{PCode, PType}
import is.hail.types.virtual.Type

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

  def asEmitParam: EmitParamType = EmitParamType(pType)

  def equalsExceptTopLevelRequiredness(that: SType): Boolean = pType.equalModuloRequired(that.pType)
}
