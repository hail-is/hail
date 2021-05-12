package is.hail.types.physical.stypes

import is.hail.annotations.Region
import is.hail.asm4s.{Code, Settable, TypeInfo, Value}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, EmitParamType, PCodeEmitParamType, PCodeParamType, SortOrder}
import is.hail.types.{TypeWithRequiredness, VirtualTypeWithReq}
import is.hail.types.physical.stypes.interfaces.SStream
import is.hail.types.physical.stypes.primitives._
import is.hail.types.physical.{PCode, PType}
import is.hail.types.virtual._


object SType {
  def chooseCompatibleType(req: VirtualTypeWithReq, stypes: SType*): SType = {
    if (stypes.toSet.size == 1)
      stypes.head
    else
      req.canonicalEmitType.st
  }

  def canonical(virt: Type): SType = {
    PType.canonical(virt).sType
  }

  def extractPrimCode(cb: EmitCodeBuilder, x: PCode): Code[_] = x.st.virtualType match {
      case TInt32 => x.asInt.intCode(cb)
      case TInt64 => x.asLong.longCode(cb)
      case TFloat32 => x.asFloat.floatCode(cb)
      case TFloat64 => x.asDouble.doubleCode(cb)
      case TBoolean => x.asBoolean.boolCode(cb)
    }

  def canonical(st: SType): SType = st.canonicalPType().sType
}

trait SType {
  def virtualType: Type

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]]

  def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = codeTupleTypes()

  def nCodes: Int = codeTupleTypes().length

  def fromSettables(settables: IndexedSeq[Settable[_]]): SSettable

  def fromCodes(codes: IndexedSeq[Code[_]]): SCode

  def canonicalPType(): PType

  def paramType: PCodeParamType = PCodeParamType(this)

  def asIdent: String = canonicalPType().asIdent

  def defaultValue: SCode = {
    fromCodes(codeTupleTypes().map(ti => ti.uninitializedValue))
  }

  def isPrimitive: Boolean = this match {
    case SInt32 | SInt64 | SFloat32 | SFloat64 | SBoolean => true
    case _ => false
  }

  def isRealizable: Boolean = !this.isInstanceOf[SStream]

  def castRename(t: Type): SType
}

case class EmitType(st: SType, required: Boolean) {
  def virtualType: Type = st.virtualType

  def paramType: PCodeEmitParamType = PCodeEmitParamType(this)

  def canonicalPType: PType = st.canonicalPType().setRequired(required)

  def equalModuloRequired(that: EmitType): Boolean = st == that.st
}