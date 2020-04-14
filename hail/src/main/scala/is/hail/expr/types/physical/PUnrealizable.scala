package is.hail.expr.types.physical

import is.hail.annotations.{CodeOrdering, Region}
import is.hail.asm4s.{Code, TypeInfo, Value}
import is.hail.expr.ir.{Ascending, Descending, EmitCodeBuilder, EmitMethodBuilder, SortOrder}

trait PUnrealizable extends PType {
  override def byteSize: Long =
    throw new UnsupportedOperationException("byteSize not defined on unrealizable types")

  override def alignment: Long =
    throw new UnsupportedOperationException("alignment not defined on unrealizable types")

  override def codeOrdering(mb: EmitMethodBuilder[_], other: PType, so: SortOrder): CodeOrdering =
    throw new UnsupportedOperationException("codeOrdering not defined on unrealizable types")

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering =
    throw new UnsupportedOperationException("codeOrdering not defined on unrealizable types")

  def copyFromType(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Long] =
    throw new UnsupportedOperationException("copyFromType not defined on unrealizable types")

  def copyFromTypeAndStackValue(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, stackValue: Code[_], deepCopy: Boolean): Code[_] =
    throw new UnsupportedOperationException("copyFromTypeAndStackValue not defined on unrealizable types")

  protected def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long =
    throw new UnsupportedOperationException("_copyFromAddress not defined on unrealizable types")

  override def copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long =
    throw new UnsupportedOperationException("copyFromAddress not defined on unrealizable types")

  def constructAtAddress(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Unit] =
    throw new UnsupportedOperationException("constructAtAddress not defined on unrealizable types")

  def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit =
    throw new UnsupportedOperationException("constructAtAddress not defined on unrealizable types")
}

trait PUnrealizableCode extends PCode {
  def code: Code[_] =
    throw new UnsupportedOperationException("code not defined on unrealizable types")

  def codeTuple(): IndexedSeq[Code[_]] =
    throw new UnsupportedOperationException("codeTuple not defined on unrealizable types")

  override def typeInfo: TypeInfo[_] =
    throw new UnsupportedOperationException("typeInfo not defined on unrealizable types")

  override def tcode[T](implicit ti: TypeInfo[T]): Code[T] =
    throw new UnsupportedOperationException("tcode not defined on unrealizable types")

  def store(mb: EmitMethodBuilder[_], r: Value[Region], dst: Code[Long]): Code[Unit] =
    throw new UnsupportedOperationException("store not defined on unrealizable types")

  override def allocateAndStore(mb: EmitMethodBuilder[_], r: Value[Region]): (Code[Unit], Code[Long]) =
    throw new UnsupportedOperationException("allocateAndStore not defined on unrealizable types")

  def memoize(cb: EmitCodeBuilder, name: String): PValue =
    throw new UnsupportedOperationException("memoize not defined on unrealizable types")

  def memoizeField(cb: EmitCodeBuilder, name: String): PValue =
    throw new UnsupportedOperationException("memoizeField not defined on unrealizable types")
}
