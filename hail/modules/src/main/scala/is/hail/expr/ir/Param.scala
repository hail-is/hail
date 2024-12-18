package is.hail.expr.ir

import is.hail.asm4s.{BooleanInfo, TypeInfo, Value}
import is.hail.types.physical.stypes.{EmitType, SType, SValue, SingleCodeType}
import is.hail.types.virtual.Type
import is.hail.utils.FastSeq

sealed trait ParamType {
  def nCodes: Int
}

case class CodeParamType(ti: TypeInfo[_]) extends ParamType {
  def nCodes: Int = 1

  override def toString: String = s"CodeParam($ti)"
}

case class SCodeParamType(st: SType) extends ParamType {
  def nCodes: Int = st.nSettables

  override def toString: String = s"SCodeParam($st, $nCodes)"
}

trait EmitParamType extends ParamType {
  def required: Boolean

  def virtualType: Type

  final lazy val valueTupleTypes: IndexedSeq[TypeInfo[_]] = {
    val ts = definedValueTupleTypes()
    if (required)
      ts
    else
      ts :+ BooleanInfo
  }

  final def nCodes: Int = valueTupleTypes.length

  protected def definedValueTupleTypes(): IndexedSeq[TypeInfo[_]]
}

case class SingleCodeEmitParamType(required: Boolean, sct: SingleCodeType) extends EmitParamType {
  def virtualType: Type = sct.virtualType

  def definedValueTupleTypes(): IndexedSeq[TypeInfo[_]] = FastSeq(sct.ti)

  override def toString: String = s"SingleCodeEmitParamType($required, $sct)"
}

case class SCodeEmitParamType(et: EmitType) extends EmitParamType {
  def required: Boolean = et.required

  def virtualType: Type = et.st.virtualType

  def definedValueTupleTypes(): IndexedSeq[TypeInfo[_]] = et.st.settableTupleTypes()
}

sealed trait Param

case class CodeParam(c: Value[_]) extends Param
case class EmitParam(ec: EmitCode) extends Param
case class SCodeParam(sc: SValue) extends Param
