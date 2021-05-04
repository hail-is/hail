package is.hail.types.physical

import is.hail.annotations.{Annotation, Region}
import is.hail.asm4s.{Code, TypeInfo, Value}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{Ascending, Descending, EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.types.physical.stypes.SCode

trait PUnrealizable extends PType {
  private def unsupported: Nothing =
    throw new UnsupportedOperationException(s"$this is not realizable")

  override def byteSize: Long = unsupported

  override def alignment: Long = unsupported

  protected[physical] def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long =
    unsupported

  override def copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long =
    unsupported

  def constructAtAddress(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Unit] =
    unsupported

  def unstagedStoreAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit =
    unsupported

  override def unstagedStoreJavaObject(annotation: Annotation, region: Region): Long =
    unsupported

  override def unstagedStoreJavaObjectAtAddress(addr: Long, annotation: Annotation, region: Region): Unit =
    unsupported

  override def loadCheapPCode(cb: EmitCodeBuilder, addr: Code[Long]): PCode = unsupported

  override def store(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): Code[Long] = unsupported

  override def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SCode, deepCopy: Boolean): Unit = unsupported

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

  def memoizeField(cb: EmitCodeBuilder, name: String): PValue = unsupported
}
