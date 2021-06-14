package is.hail.types.physical.stypes

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCode, EmitCodeBuilder, EmitSettable, PCodeEmitParamType, PCodeParamType}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical.PType
import is.hail.types.physical.stypes.interfaces.SStream
import is.hail.types.physical.stypes.primitives._
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

  def extractPrimCode(cb: EmitCodeBuilder, x: SCode): Code[_] = x.st.virtualType match {
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

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]]

  def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = codeTupleTypes()

  lazy val nCodes: Int = codeTupleTypes().length

  lazy val nSettables: Int = settableTupleTypes().length

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

  lazy val codeTupleTypes: IndexedSeq[TypeInfo[_]] = {
    val tc = st.codeTupleTypes()
    if (required)
      tc
    else
      tc :+ BooleanInfo
  }

  lazy val settableTupleTypes: IndexedSeq[TypeInfo[_]] = {
    val tc = st.settableTupleTypes()
    if (required)
      tc
    else
      tc :+ BooleanInfo
  }

  def fromCodes(codes: IndexedSeq[Code[_]]): EmitCode = {
    val scode = st.fromCodes(codes.take(st.nCodes))
    val m: Code[Boolean] = if (required) const(false) else coerce[Boolean](codes.last)
    val ec = EmitCode(Code._empty, m, scode)
    if (ec.required && !this.required)
      ec.setOptional
    else
      ec
  }

  def fromSettables(settables: IndexedSeq[Settable[_]]): EmitSettable = new EmitSettable(
    if (required) None else Some(coerce[Boolean](settables.last)),
    st.fromSettables(settables.take(st.nSettables))
  )

  def nCodes: Int = codeTupleTypes.length

  def nSettables: Int = settableTupleTypes.length
}