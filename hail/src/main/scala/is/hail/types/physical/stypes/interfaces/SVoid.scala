package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s.{Settable, TypeInfo, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.TypeWithRequiredness
import is.hail.types.physical.PType
import is.hail.types.physical.stypes._
import is.hail.types.virtual.{TVoid, Type}
import is.hail.utils.FastSeq

case object SVoid extends SType {

  override def virtualType: Type = TVoid

  override def castRename(t: Type): SType = this

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue = value

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = IndexedSeq()

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SSettable = throw new UnsupportedOperationException

  override def fromValues(values: IndexedSeq[Value[_]]): SValue = throw new UnsupportedOperationException

  override def storageType(): PType = throw new UnsupportedOperationException

  override def copiedType: SType = this

  override def _typeWithRequiredness: TypeWithRequiredness = throw new UnsupportedOperationException

  override def containsPointers: Boolean = false
}

case object SVoidValue extends SValue with SUnrealizableValue {
  self =>

  override def st: SType = SVoid

  override def valueTuple: IndexedSeq[Value[_]] = FastSeq()
}
