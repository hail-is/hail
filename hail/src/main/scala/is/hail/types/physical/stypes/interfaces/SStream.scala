package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s.{Code, Settable, TypeInfo, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.expr.ir.streams.StreamProducer
import is.hail.types.physical.stypes.{EmitType, SCode, SSettable, SType, SUnrealizableCode, SValue}
import is.hail.types.physical.{PCanonicalStream, PStream, PType}
import is.hail.types.virtual.{TStream, Type}

case class SStream(elementEmitType: EmitType) extends SType {
  def elementType: SType = elementEmitType.st

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    if (deepCopy) throw new UnsupportedOperationException

    assert(value.st == this)
    value
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = throw new UnsupportedOperationException

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode = throw new UnsupportedOperationException

  def fromCodes(codes: IndexedSeq[Code[_]]): SCode = throw new UnsupportedOperationException

  def fromSettables(settables: IndexedSeq[Settable[_]]): SSettable = throw new UnsupportedOperationException

  def canonicalPType(): PType = PCanonicalStream(elementEmitType.canonicalPType)

  override def virtualType: Type = TStream(elementType.virtualType)

  override def castRename(t: Type): SType = ???
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
