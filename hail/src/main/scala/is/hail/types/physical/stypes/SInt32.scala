package is.hail.types.physical.stypes

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, IntInfo, Settable, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.mtypes.{MInt32, MValue}

case object SInt32 extends SType {
  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], mv: MValue): SCode = {
    mv.typ match {
      case MInt32 => new SInt32Code(Region.loadInt(mv.addr))
    }
  }

  def coerceOrCopySValue(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deep: Boolean): SCode = {
    assert(value.typ == this)
    value
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = {
    IndexedSeq(IntInfo)
  }

  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = {
    ???
  }
}

class SInt32Code(x: Code[Int]) extends SCode {
  def typ: SType = SInt32

  override def memoize(cb: EmitCodeBuilder): SValue = {
    val xvar = cb.newLocal[Int]("sint32_memoize")
    cb.assign(xvar, x)
    new SInt32Value(xvar)
  }

  def intCode: Code[Int] = x
}

class SInt32Value(x: Settable[Int]) extends SValue {
  def typ: SType = SInt32

  def intValue: Value[Int] = x
}