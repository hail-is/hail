package is.hail.types.physical.stypes.interfaces

import is.hail.annotations.Region
import is.hail.asm4s.{Code, Settable, TypeInfo, UnitInfo, Value}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.stypes.{SCode, SSettable, SType}
import is.hail.types.physical.{PCode, PType, PUnrealizableCode, PValue, PVoid}
import is.hail.types.virtual.{TVoid, Type}

case object SVoid extends SType {

  override def virtualType: Type = TVoid

  override def castRename(t: Type): SType = this

  def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): SCode = value

  def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = IndexedSeq()

  def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): SCode = throw new UnsupportedOperationException

  def fromCodes(codes: IndexedSeq[Code[_]]): SCode = throw new UnsupportedOperationException

  def fromSettables(settables: IndexedSeq[Settable[_]]): SSettable = throw new UnsupportedOperationException

  def canonicalPType(): PType = throw new UnsupportedOperationException
}

case object PVoidCode extends PCode with PUnrealizableCode {
  self =>

  override def st: SType = SVoid

  override def code: Code[_] = Code._empty

  def memoize(cb: EmitCodeBuilder, name: String): PValue = new PValue {
    val pt: PType = PVoid
    val st: SType = SVoid

    def get: PCode = self
  }
}
