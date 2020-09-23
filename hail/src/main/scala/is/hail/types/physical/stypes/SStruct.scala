package is.hail.types.physical.stypes
import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.Value
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.mtypes.{MType, MValue}

trait SStruct extends SType {
  def baseToMCanonicalValue(cb: EmitCodeBuilder, region: Value[Region], value: SCode): MValue = {
    val mt = MType.canonical(value.typ)
    val umv = mt.allocate(cb, region)
    umv.store(cb, region, value.memoize(cb))
  }

  def baseCoerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deep: Boolean): SCode = {
    // this should work on any mtype/stype combination, but in practice will rarely get called
    // instead, we'll write match statements on types to generate efficient code for specific coerce

    // FIXME: currently doesn't do deep copy correctly

    loadFrom(cb, region, baseToMCanonicalValue(cb, region, value))
  }

  def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = {
    //
    ???
  }
}

trait SStructCode extends SCode {
  def structType: SStruct

  override def typ: SType = structType
  override def memoize(cb: EmitCodeBuilder): SStructValue
}

trait SStructValue extends SValue {
  def structType: SStruct

  final def typ: SType = structType

  def loadField(cb: EmitCodeBuilder, region: Value[Region], idx: Int): IEmitSCode
}
