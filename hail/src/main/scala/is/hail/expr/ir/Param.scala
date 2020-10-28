package is.hail.expr.ir

import is.hail.asm4s.{Code, TypeInfo}
import is.hail.types.physical.{PCode, PType}

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

case class EmitParamType(pt: PType) extends ParamType {
  def nCodes: Int = pt.nCodes

  override def toString: String = s"EmitParam($pt, $nCodes)"
}

sealed trait Param

case class CodeParam(c: Code[_]) extends Param
case class EmitParam(ec: EmitCode) extends Param
case class PCodeParam(pc: PCode) extends Param
