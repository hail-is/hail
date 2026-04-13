package is.hail.expr.ir

import is.hail.asm4s.{BooleanInfo, TypeInfo, Value}
import is.hail.collection.FastSeq
import is.hail.types.physical.stypes.{EmitType, SType, SValue, SingleCodeType}
import is.hail.types.virtual.Type

sealed trait ParamType {
  def codeTypes: IndexedSeq[TypeInfo[_]]

  def nCodes: Int = codeTypes.length
}

case class CodeParamType(ti: TypeInfo[_]) extends ParamType {
  override def codeTypes: IndexedSeq[TypeInfo[_]] = FastSeq(ti)

  override def toString: String = s"CodeParam($ti)"
}

case class SCodeParamType(st: SType) extends ParamType {
  override def codeTypes: IndexedSeq[TypeInfo[_]] = st.settableTupleTypes()

  override def toString: String = s"SCodeParam($st, $nCodes)"
}

trait EmitParamType extends ParamType {
  def required: Boolean

  def virtualType: Type

  final override lazy val codeTypes: IndexedSeq[TypeInfo[_]] = {
    val ts = definedValueTupleTypes()
    if (required)
      ts
    else
      ts :+ BooleanInfo
  }

  protected def definedValueTupleTypes(): IndexedSeq[TypeInfo[_]]
}

case class SingleCodeEmitParamType(required: Boolean, sct: SingleCodeType) extends EmitParamType {
  override def virtualType: Type = sct.virtualType

  override def definedValueTupleTypes(): IndexedSeq[TypeInfo[_]] = FastSeq(sct.ti)

  override def toString: String = s"SingleCodeEmitParamType($required, $sct)"
}

case class SCodeEmitParamType(et: EmitType) extends EmitParamType {
  override def required: Boolean = et.required

  override def virtualType: Type = et.st.virtualType

  override def definedValueTupleTypes(): IndexedSeq[TypeInfo[_]] = et.st.settableTupleTypes()
}

sealed trait Param

case class CodeParam(c: Value[_]) extends Param
case class EmitParam(ec: EmitCode) extends Param
case class SCodeParam(sc: SValue) extends Param
