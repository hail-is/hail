package is.hail.types.physical.stypes

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{CodeLabel, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, IEmitCode, SortOrder}
import is.hail.types.BaseType
import is.hail.types.physical.mtypes.{MCode, MInt32, MStruct, MType, MValue}

case class IEmitSCode(Lmissing: CodeLabel, Lpresent: CodeLabel, pc: SCode) {
  def memoize(cb: EmitCodeBuilder): EmitSValue = ???
}

case class EmitSValue(missing: Boolean, value: SValue) {
  def toI: IEmitSCode = ???
}

// replaces the current PCode
trait SCode {
  def typ: SType

  def memoize(cb: EmitCodeBuilder): SValue

  def copyToRegion(cb: EmitCodeBuilder, region: Value[Region]): SCode = typ.coerceOrCopySValue(cb, region, this, deep = true)

  def castTo(cb: EmitCodeBuilder, region: Value[Region], newTyp: SType): SCode = newTyp.coerceOrCopySValue(cb, region, this, deep = false)

  def asStruct: SStructCode = asInstanceOf[SStructCode]
  def asString: SStringCode = asInstanceOf[SStringCode]

}

// replaces the current PValue
trait SValue {
  def typ: SType

  def asStruct: SStructValue = asInstanceOf[SStructValue]
  def asString: SStringValue = asInstanceOf[SStringValue]
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