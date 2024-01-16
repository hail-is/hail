package is.hail.types.physical.stypes

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{
  EmitCodeBuilder, EmitSettable, EmitValue, SCodeEmitParamType, SCodeParamType,
}
import is.hail.types.{TypeWithRequiredness, VirtualTypeWithReq}
import is.hail.types.physical.PType
import is.hail.types.physical.stypes.concrete.SUnreachable
import is.hail.types.physical.stypes.interfaces.SStream
import is.hail.types.physical.stypes.primitives._
import is.hail.types.virtual._

object SType {
  def chooseCompatibleType(req: VirtualTypeWithReq, stypes: SType*): SType = {
    val reachable = stypes.filter(t => !t.isInstanceOf[SUnreachable]).toSet

    // all unreachable
    if (reachable.isEmpty)
      SUnreachable.fromVirtualType(req.t)
    else if (reachable.size == 1) // only one reachable stype
      reachable.head
    else
      req.canonicalEmitType.st // fall back to canonical emit type from requiredness
  }

  def canonical(virt: Type): SType =
    PType.canonical(virt).sType

  def extractPrimValue(cb: EmitCodeBuilder, x: SValue): Value[_] = x.st.virtualType match {
    case TInt32 => x.asInt.value
    case TInt64 => x.asLong.value
    case TFloat32 => x.asFloat.value
    case TFloat64 => x.asDouble.value
    case TBoolean => x.asBoolean.value
  }
}

trait SType {
  def virtualType: Type

  final def coerceOrCopy(
    cb: EmitCodeBuilder,
    region: Value[Region],
    value: SValue,
    deepCopy: Boolean,
  ): SValue =
    value.st match {
      case _: SUnreachable => this.defaultValue
      case _ => _coerceOrCopy(cb, region, value, deepCopy)
    }

  protected[stypes] def _coerceOrCopy(
    cb: EmitCodeBuilder,
    region: Value[Region],
    value: SValue,
    deepCopy: Boolean,
  ): SValue

  def settableTupleTypes(): IndexedSeq[TypeInfo[_]]

  lazy val nSettables: Int = settableTupleTypes().length

  def fromSettables(settables: IndexedSeq[Settable[_]]): SSettable

  def fromValues(values: IndexedSeq[Value[_]]): SValue

  def storageType(): PType

  def copiedType: SType

  def paramType: SCodeParamType = SCodeParamType(this)

  def asIdent: String = getClass.getSimpleName

  def defaultValue: SValue =
    fromValues(settableTupleTypes().map(ti => ti.uninitializedValue))

  def isPrimitive: Boolean = this match {
    case SInt32 | SInt64 | SFloat32 | SFloat64 | SBoolean => true
    case _ => false
  }

  def isRealizable: Boolean = !this.isInstanceOf[SStream]

  def castRename(t: Type): SType

  protected[stypes] def _typeWithRequiredness: TypeWithRequiredness

  final def typeWithRequiredness: VirtualTypeWithReq =
    VirtualTypeWithReq(virtualType, _typeWithRequiredness)

  def containsPointers: Boolean
}

case class EmitType(st: SType, required: Boolean) {
  def virtualType: Type = st.virtualType

  def paramType: SCodeEmitParamType = SCodeEmitParamType(this)

  def storageType: PType = st.storageType().setRequired(required)

  def copiedType: EmitType = copy(st = st.copiedType)

  def typeWithRequiredness: VirtualTypeWithReq = st.typeWithRequiredness.setRequired(required)

  def equalModuloRequired(that: EmitType): Boolean = st == that.st

  lazy val settableTupleTypes: IndexedSeq[TypeInfo[_]] = {
    val tc = st.settableTupleTypes()
    if (required)
      tc
    else
      tc :+ BooleanInfo
  }

  def fromSettables(settables: IndexedSeq[Settable[_]]): EmitSettable = new EmitSettable(
    if (required) None else Some(coerce[Boolean](settables.last)),
    st.fromSettables(settables.take(st.nSettables)),
  )

  def fromValues(values: IndexedSeq[Value[_]]): EmitValue = EmitValue(
    if (required) None else Some(coerce[Boolean](values.last)),
    st.fromValues(values.take(st.nSettables)),
  )

  def nSettables: Int = settableTupleTypes.length

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: EmitValue, deepCopy: Boolean)
    : EmitValue = {
    if (value.emitType == this && (!deepCopy || !value.st.containsPointers))
      value
    else
      (required, value.required) match {
        case (true, _) => EmitValue.present(st.coerceOrCopy(cb, region, value.get(cb), deepCopy))
        case (false, true) =>
          EmitValue.present(st.coerceOrCopy(cb, region, value.get(cb), deepCopy)).setOptional
        case (false, false) =>
          cb.memoize(value.toI(cb).map(cb)(value => st.coerceOrCopy(cb, region, value, deepCopy)))
      }
  }
}
