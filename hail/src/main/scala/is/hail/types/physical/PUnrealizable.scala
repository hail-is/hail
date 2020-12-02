package is.hail.types.physical

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, LineNumber, TypeInfo, Value}
import is.hail.expr.ir.{Ascending, Descending, EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.stypes.SCode

trait PUnrealizable extends PType {
  private def unsupported: Nothing =
    throw new UnsupportedOperationException(s"$this is not realizable")

  override def byteSize: Long = unsupported

  override def alignment: Long = unsupported

  override def codeOrdering(mb: EmitMethodBuilder[_], other: PType, so: SortOrder): CodeOrdering =
    unsupported

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering =
    unsupported

  protected[physical] def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long =
    unsupported

  override def copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long =
    unsupported

  def unstagedStoreAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit =
    unsupported

  override def loadCheapPCode(cb: EmitCodeBuilder, addr: Code[Long])(implicit line: LineNumber): PCode = unsupported

  override def store(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean)(implicit line: LineNumber): Code[Long] = unsupported

  override def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SCode, deepCopy: Boolean)(implicit line: LineNumber): Unit = unsupported

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

  def memoizeField(cb: EmitCodeBuilder, name: String)(implicit line: LineNumber): PValue = unsupported
}
