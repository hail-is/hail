package is.hail.expr.ir

import is.hail.asm4s.{BooleanInfo, Code, TypeInfo, classInfo}
import is.hail.types.physical.{PCode, PType, SingleCodePCode, SingleCodeType}
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

case class PCodeParamType(pt: PType) extends ParamType {
  def nCodes: Int = pt.nCodes

  override def toString: String = s"PCodeParam($pt, $nCodes)"
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

case class PCodeEmitParamType(pt: PType) extends EmitParamType {
  def required: Boolean = pt.required

  def virtualType: Type = pt.virtualType

  def definedTupleTypes(): IndexedSeq[TypeInfo[_]] = pt.codeTupleTypes()

  override def toString: String = s"PTypeEmitParamType($pt, $nCodes)"
}

sealed trait Param

case class CodeParam(c: Code[_]) extends Param
case class EmitParam(ec: EmitCode) extends Param
case class PCodeParam(pc: PCode) extends Param
