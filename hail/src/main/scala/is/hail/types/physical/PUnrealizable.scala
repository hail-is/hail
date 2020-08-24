package is.hail.types.physical

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, TypeInfo, Value}
import is.hail.expr.ir.{Ascending, Descending, EmitCodeBuilder, EmitMethodBuilder, SortOrder}

trait PUnrealizable extends PType {
  private def unsupported: Nothing =
    throw new UnsupportedOperationException(s"$this is not realizable")

  override def byteSize: Long = unsupported

  override def alignment: Long = unsupported

  override def codeOrdering(mb: EmitMethodBuilder[_], other: PType, so: SortOrder): CodeOrdering =
    unsupported

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering =
    unsupported

  def copyFromType(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Long] =
    unsupported

  def copyFromTypeAndStackValue(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, stackValue: Code[_], deepCopy: Boolean): Code[_] =
    unsupported

  protected def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long =
    unsupported

  override def copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long =
    unsupported

  def constructAtAddress(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Unit] =
    unsupported

  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit =
    unsupported

  def stackValueToAnnotation(c: Code[_]): Code[AnyRef] = unsupported

  override def encodableType: PType = unsupported

  override def containsPointers: Boolean = {
    throw new UnsupportedOperationException("containsPointers not supported on PUnrealizable")
  }
}

trait PUnrealizableCode extends PCode {
  private def unsupported: Nothing =
    throw new UnsupportedOperationException(s"$pt is not realizable")

  def code: Code[_] = unsupported

  def codeTuple(): IndexedSeq[Code[_]] = unsupported

  override def typeInfo: TypeInfo[_] = unsupported

  override def tcode[T](implicit ti: TypeInfo[T]): Code[T] = unsupported

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] =
    unsupported

  override def allocateAndStore(mb: EmitMethodBuilder[_], r: Value[Region]): (Code[Unit], Code[Long]) =
    unsupported

  def memoizeField(cb: EmitCodeBuilder, name: String): PValue = unsupported

  override def castTo(mb: EmitMethodBuilder[_], region: Value[Region], destPtype: PType, deepCopy: Boolean = false): PCode = {
    assert(destPtype == pt)
    this
  }
}
