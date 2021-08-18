package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s.{Code, Settable, TypeInfo, UnitInfo, Value}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.{RPrimitive, TypeWithRequiredness}
import is.hail.types.physical.stypes.{SCode, SSettable, SType, SUnrealizableCode, SValue}
import is.hail.types.physical.{PType, PVoid}
import is.hail.types.virtual.{TVoid, Type}

case object SVoid extends SType {

  override def virtualType: Type = TVoid

  override def castRename(t: Type): SType = this

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = value

  override def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = IndexedSeq()

  override def fromCodes(codes: IndexedSeq[Code[_]]): SCode = throw new UnsupportedOperationException

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SSettable = throw new UnsupportedOperationException

  override def fromValues(values: IndexedSeq[Value[_]]): SValue = throw new UnsupportedOperationException

  override def storageType(): PType = throw new UnsupportedOperationException

  override def copiedType: SType = this

  override def _typeWithRequiredness: TypeWithRequiredness = throw new UnsupportedOperationException

  override def containsPointers: Boolean = false
}

case object SVoidCode extends SCode with SUnrealizableCode {
  self =>

  override def st: SType = SVoid

  override def code: Code[_] = Code._empty

  def memoize(cb: EmitCodeBuilder, name: String): SValue = new SValue {
    val pt: PType = PVoid
    val st: SType = SVoid

    def get: SCode = self
  }
}
