package is.hail.types.physical.stypes

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, TypeInfo, UnitInfo, Value}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.{PCode, PType, PUnrealizableCode, PValue, PVoid}

case object SVoid extends SType {

  override def pType: PType = PVoid

  override def coerceOrCopy(cb: EmitCodeBuilder, region: Value[Region], value: PCode, deepCopy: Boolean): PCode = value

  override def codeOrdering(mb: EmitMethodBuilder[_], other: SType, so: SortOrder): CodeOrdering = throw new UnsupportedOperationException

  override def codeTupleTypes(): IndexedSeq[TypeInfo[_]] = IndexedSeq()

  override def loadFrom(cb: EmitCodeBuilder, region: Value[Region], pt: PType, addr: Code[Long]): PCode = throw new UnsupportedOperationException
}

case object PVoidCode extends PCode with PUnrealizableCode {
  self =>

  override def pt: PType = PVoid

  override def st: SType = SVoid

  override def typeInfo: TypeInfo[_] = UnitInfo

  override def code: Code[_] = Code._empty

  override def tcode[T](implicit ti: TypeInfo[T]): Code[T] = {
    assert(ti == typeInfo)
    code.asInstanceOf[Code[T]]
  }

  def memoize(cb: EmitCodeBuilder, name: String): PValue = new PValue {
    val pt: PType = PVoid
    val st: SType = SVoid

    def get: PCode = self
  }
}
