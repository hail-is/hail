package is.hail.types.physical.stypes

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{CodeLabel, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.BaseType
import is.hail.types.physical.mtypes.{MCode, MInt32, MStruct, MType, MValue}

case class IEmitSCode(Lmissing: CodeLabel, Lpresent: CodeLabel, pc: SCode) {
  def memoize(cb: EmitCodeBuilder): EmitSValue = ???

  def consume[T](cb: EmitCodeBuilder, ifMissing: => Unit, ifPresent: (SCode) => T): T = {
    val Lafter = CodeLabel()
    cb.define(Lmissing)
    ifMissing
    if (cb.isOpenEnded) cb.goto(Lafter)
    cb.define(Lpresent)
    val ret = ifPresent(pc)
    cb.define(Lafter)
    ret
  }

  def map(cb: EmitCodeBuilder)(f: (SCode) => SCode): IEmitSCode = {
    val Lpresent2 = CodeLabel()
    cb.define(Lpresent)
    val pc2 = f(pc)
    cb.goto(Lpresent2)
    IEmitSCode(Lmissing, Lpresent2, pc2)
  }

  def mapMissing(cb: EmitCodeBuilder)(ifMissing: => Unit): IEmitSCode = {
    val Lmissing2 = CodeLabel()
    cb.define(Lmissing)
    ifMissing
    cb.goto(Lmissing2)
    IEmitSCode(Lmissing2, Lpresent, pc)
  }

  def flatMap(cb: EmitCodeBuilder)(f: (SCode) => IEmitSCode): IEmitSCode = {
    cb.define(Lpresent)
    val ec2 = f(pc)
    cb.define(ec2.Lmissing)
    cb.goto(Lmissing)
    IEmitSCode(Lmissing, ec2.Lpresent, ec2.pc)
  }

  def handle(cb: EmitCodeBuilder, ifMissing: => Unit): SCode = {
    cb.define(Lmissing)
    ifMissing
    cb.define(Lpresent)
    pc
  }

}

case class EmitSValue(missing: Boolean, value: SValue) {
  def toI: IEmitSCode = ???
}

// replaces the current SCode
trait SCode {
  def typ: SType

  def memoize(cb: EmitCodeBuilder): SValue

  def copyToRegion(cb: EmitCodeBuilder, region: Value[Region]): SCode = typ.coerceOrCopySValue(cb, region, this, deep = true)

  def castTo(cb: EmitCodeBuilder, region: Value[Region], newTyp: SType): SCode = newTyp.coerceOrCopySValue(cb, region, this, deep = false)

  def asStruct: SStructCode = asInstanceOf[SStructCode]
  def asString: SStringCode = asInstanceOf[SStringCode]
  def asInt32: SInt32Code = asInstanceOf[SInt32Code]
  def asBoolean: SBooleanCode = asInstanceOf[SBooleanCode]

}

// replaces the current PValue
trait SValue {
  def typ: SType

  def asStruct: SStructValue = asInstanceOf[SStructValue]
  def asString: SStringValue = asInstanceOf[SStringValue]
  def asInt32: SInt32Value = asInstanceOf[SInt32Value]
  def asBoolean: SBooleanValue = asInstanceOf[SBooleanValue]
}

trait SPointer extends SType {
  def mType: MType
}

trait SType {
  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], mv: MValue): SCode

  def coerceOrCopySValue(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deep: Boolean): SCode

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]]

  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering
}

// how do we implement MakeArray(makeStructs) to allocate once?