package is.hail.expr.ir

import is.hail.asm4s.{BooleanInfo, Code, TypeInfo, classInfo}
import is.hail.types.physical.stypes.{EmitType, SCode, SType}
import is.hail.types.physical.SingleCodeType
import is.hail.types.virtual.Type
import is.hail.utils.FastIndexedSeq

import scala.language.existentials

sealed trait ParamType  {
  def nCodes: Int
}

case class CodeParamType(ti: TypeInfo[_]) extends ParamType {
  def nCodes: Int = 1

  override def toString: String = s"CodeParam($ti)"
}

case class PCodeParamType(st: SType) extends ParamType {
  def nCodes: Int = st.nCodes

  override def toString: String = s"PCodeParam($st, $nCodes)"
}

trait EmitParamType extends ParamType {
  def required: Boolean

  def virtualType: Type

  final lazy val codeTupleTypes: IndexedSeq[TypeInfo[_]] = {
    val ts = definedTupleTypes()
    if (required)
      ts
    else
      ts :+ BooleanInfo
  }

  final def nCodes: Int = codeTupleTypes.length

  protected def definedTupleTypes(): IndexedSeq[TypeInfo[_]]
}

case class SingleCodeEmitParamType(required: Boolean, sct: SingleCodeType) extends EmitParamType {
  def virtualType: Type = sct.virtualType

  def definedTupleTypes(): IndexedSeq[TypeInfo[_]] = FastIndexedSeq(sct.ti)

  override def toString: String = s"SingleCodeEmitParamType($required, $sct)"
}

case class PCodeEmitParamType(et: EmitType) extends EmitParamType {
  def required: Boolean = et.required

  def virtualType: Type = et.st.virtualType

  def definedTupleTypes(): IndexedSeq[TypeInfo[_]] = et.st.codeTupleTypes()
}

sealed trait Param

case class CodeParam(c: Code[_]) extends Param
case class EmitParam(ec: EmitCode) extends Param
case class PCodeParam(pc: SCode) extends Param
