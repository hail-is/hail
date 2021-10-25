package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s.{Settable, TypeInfo, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.expr.ir.streams.StreamProducer
import is.hail.types.physical.PType
import is.hail.types.physical.stypes._
import is.hail.types.virtual.{TStream, Type}
import is.hail.types.{RIterable, TypeWithRequiredness}

final case class SStream(elementEmitType: EmitType) extends SType {
  def elementType: SType = elementEmitType.st

  override def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): SValue = {
    if (deepCopy) throw new NotImplementedError()

    assert(value.st == this)
    value
  }

  override def settableTupleTypes(): IndexedSeq[TypeInfo[_]] = throw new NotImplementedError()

  override def fromSettables(settables: IndexedSeq[Settable[_]]): SSettable = throw new NotImplementedError()

  override def fromValues(values: IndexedSeq[Value[_]]): SValue = throw new NotImplementedError()

  override def storageType(): PType = throw new NotImplementedError()

  override def copiedType: SType = throw new NotImplementedError()

  override def containsPointers: Boolean = throw new NotImplementedError()

  override def virtualType: Type = TStream(elementType.virtualType)

  override def castRename(t: Type): SType = throw new UnsupportedOperationException("rename on stream")

  override def _typeWithRequiredness: TypeWithRequiredness = RIterable(elementEmitType.typeWithRequiredness.r)
}

object SStreamValue{
  def apply(producer: StreamProducer): SStreamValue = SStreamValue(SStream(producer.element.emitType), producer)
}

final case class SStreamValue(st: SStream, producer: StreamProducer) extends SUnrealizableValue {
  def valueTuple: IndexedSeq[Value[_]] = throw new NotImplementedError()
}
