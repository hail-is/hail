package is.hail.types.physical

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.BaseType

class UninitializedMValue(val addr: Value[Long], val typ: MType) {
  final def store(cb: EmitCodeBuilder, region: Value[Region], code: SCode): MValue = {
    store(cb, region, code.memoize(cb))
  }

  final def store(cb: EmitCodeBuilder, region: Value[Region], value: SValue): MValue = {
    typ.storeFromSValue(cb, this, value)
    new MValue(addr, typ)
  }
}

class MCode(val addr: Code[Long], val typ: MType) {
  def memoize(cb: EmitCodeBuilder): MValue = {
    val x = cb.newLocal[Long]("mcode_memoize")
    cb.assign(x, addr)
    new MValue(x, typ)
  }
}

class MValue(val addr: Value[Long], val typ: MType)

trait MType extends BaseType {
  def byteSize: Long

  def alignment: Long

  def allocate(cb: EmitCodeBuilder, region: Value[Region]): UninitializedMValue

  def storeFromSValue(cb: EmitCodeBuilder, memory: UninitializedMValue, value: SValue): Unit

  def storeFromMValue(cb: EmitCodeBuilder, memory: UninitializedMValue, value: MValue): Unit

  def coerceOrCopyMValue(cb: EmitCodeBuilder, region: Value[Region], value: MValue, deep: Boolean): MValue
}

// replaces the current PCode
trait SCode {
  def typ: SType

  def memoize(cb: EmitCodeBuilder): SValue

  def copyToRegion(cb: EmitCodeBuilder, region: Value[Region]): SCode = typ.coerceOrCopySValue(cb, region, this, deep = true)

  def castTo(cb: EmitCodeBuilder, region: Value[Region], newTyp: SType): SCode = newTyp.coerceOrCopySValue(cb, region, this, deep = false)
}

// replaces the current PValue
trait SValue {
  def typ: SType
}

trait SType extends BaseType {
  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], mv: MValue): SValue

  def coerceOrCopySValue(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deep: Boolean): SCode

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]]

  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering
}

// how do we implement MakeArray(makeStructs) to allocate once?