package is.hail.types.physical.stypes

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{BooleanInfo, Code, IntInfo, Settable, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.mtypes.{MInt32, MValue}

case object SBoolean extends SType {
  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], mv: MValue): SCode = {
    mv.typ match {
      case _ => new SBooleanCode(Region.loadBoolean(mv.addr))
    }
  }

  def coerceOrCopySValue(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deep: Boolean): SCode = {
    assert(value.typ == this)
    value
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = {
    IndexedSeq(BooleanInfo)
  }

  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = {
    ???
  }
}

class SBooleanCode(x: Code[Boolean]) extends SCode {
  def typ: SType = SBoolean

  override def memoize(cb: EmitCodeBuilder): SValue = {
    val xvar = cb.newLocal[Boolean]("sboolean_memoize")
    cb.assign(xvar, x)
    new SBooleanValue(xvar)
  }

  def boolCode: Code[Boolean] = x
}

class SBooleanValue(x: Settable[Boolean]) extends SValue {
  def typ: SType = SBoolean

  def boolValue: Value[Boolean] = x
}