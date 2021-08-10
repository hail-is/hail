package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s.{Code, Settable, TypeInfo, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.expr.ir.streams.StreamProducer
import is.hail.types.{RIterable, TypeWithRequiredness}
import is.hail.types.physical.stypes.{EmitType, SCode, SSettable, SType, SUnrealizableCode, SValue}
import is.hail.types.physical.PType
import is.hail.types.virtual.{TStream, Type}

case class SStream(elementEmitType: EmitType) extends SType {
  def elementType: SType = elementEmitType.st

  def _coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    if (deepCopy) throw new NotImplementedError()

    assert(value.st == this)
    value
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = throw new NotImplementedError()

  def fromCodes(codes: IndexedSeq[Code[_]]): SCode = throw new NotImplementedError()

  def fromSettables(settables: IndexedSeq[Settable[_]]): SSettable = throw new NotImplementedError()

  def storageType(): PType = throw new NotImplementedError()

  def copiedType: SType = throw new NotImplementedError()

  override def containsPointers: Boolean = throw new NotImplementedError()

  override def virtualType: Type = TStream(elementType.virtualType)

  override def castRename(t: Type): SType = throw new UnsupportedOperationException("rename on stream")

  override def _typeWithRequiredness: TypeWithRequiredness = RIterable(elementEmitType.typeWithRequiredness.r)
}

object SStreamCode{
  def apply(producer: StreamProducer): SStreamCode = SStreamCode(SStream(producer.element.emitType), producer)
}

final case class SStreamCode(st: SStream, producer: StreamProducer) extends SCode with SUnrealizableCode {
  self =>
  def memoize(cb: EmitCodeBuilder, name: String): SValue = new SValue {

    override def st: SType = self.st

    var used: Boolean = false

    def get: SCode = {
      assert(!used)
      used = true
      self
    }
  }
}
