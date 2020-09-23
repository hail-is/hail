package is.hail.types.physical.stypes
import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{LongInfo, TypeInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.mtypes.{MStruct, MValue}

case class SStructPointer(mType: MStruct) extends SStruct with SPointer {

  override def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = IndexedSeq(LongInfo)

  override def loadFrom(cb: EmitCodeBuilder, region: Value[Region], mv: MValue): SCode = SStructPointerValue(this, mv)

  override def coerceOrCopySValue(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deep: Boolean): SCode = {
    ???
  }
}

case class SStructPointerValue(structType: SStructPointer, mv: MValue) extends SStructCode with SStructValue {
  override def memoize(cb: EmitCodeBuilder): SStructValue = this

  override def loadField(cb: EmitCodeBuilder, region: Value[Region], idx: Int): IEmitSCode = {
    structType.mType.getField(cb, idx, mv)
      .mapS(cb) { fieldMCode =>
        structType.mType.fields(idx).typ
          .pointerType
          .loadFrom(cb, region, fieldMCode.memoize(cb))
      }
  }
}

case class  SStructPointerValue(structType: SStructPointer, mv: ) extends SStructValue {

}
