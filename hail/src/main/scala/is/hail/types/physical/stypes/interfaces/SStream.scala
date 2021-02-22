package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s.{Code, Settable, TypeInfo, Value}
import is.hail.expr.ir.EmitStream.SizedStream
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.stypes.{SCode, SSettable, SType}
import is.hail.types.physical.{PCanonicalStream, PCode, PStream, PStreamCode, PType, PValue}

case class SStream(elementType: SType, separateRegions: Boolean = false) extends SType {
  def pType: PStream = PCanonicalStream(elementType.pType, separateRegions, false)

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = {
    if (deepCopy) throw new UnsupportedOperationException

    assert(value.st == this)
    value
  }

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = throw new UnsupportedOperationException

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode = throw new UnsupportedOperationException

  def fromCodes(codes: IndexedSeq[Code[_]]): SCode = throw new UnsupportedOperationException

  def fromSettables(settables: IndexedSeq[Settable[_]]): SSettable = throw new UnsupportedOperationException

  def canonicalPType(): PType = pType
}


final case class SStreamCode(st: SStream, stream: SizedStream) extends PStreamCode {
  self =>
  override def pt: PStream = st.pType

  def memoize(cb: EmitCodeBuilder, name: String): PValue = new PValue {
    def pt: PStream = PCanonicalStream(st.pType)

    override def st: SType = self.st

    var used: Boolean = false

    def get: PCode = {
      assert(!used)
      used = true
      self
    }
  }
}
