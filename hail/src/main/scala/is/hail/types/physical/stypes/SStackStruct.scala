package is.hail.types.physical.stypes

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.mtypes.{MStruct, MValue}

case class SStackStructField[T](v: T, name: String, idx: Int)

case class SStackStruct(fields: IndexedSeq[SStackStructField[SType]]) extends SStruct {
  override def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = ???

  override def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = fields.flatMap(_.v.codeTupleTypes())

  override def loadFrom(cb: EmitCodeBuilder, region: Value[Region], mv: MValue): SCode = {
    val mvS = mv.typ.asInstanceOf[MStruct]
    SStackStructCode(this,
      fields.map(f => mvS.getField(cb, f.idx, mv).mapS(cb) { fieldMC =>
        val fieldMV = fieldMC.memoize(cb)
        f.v.loadFrom(cb, region, fieldMV)
      })
    )
  }

  override def coerceOrCopySValue(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deep: Boolean): SCode = {
    ???
  }
}

case class SStackStructCode(structType: SStackStruct, values: IndexedSeq[IEmitSCode]) extends SStructCode {
  override def memoize(cb: EmitCodeBuilder): SStructValue = {
    SStackStructValue(structType, values.map(_.memoize(cb)))
  }
}

case class SStackStructValue(structType: SStackStruct, values: IndexedSeq[EmitSValue]) extends SStructValue {
  def loadField(idx: Int): IEmitSCode = values(idx).toI
}