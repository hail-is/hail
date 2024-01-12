package is.hail.types.physical

import is.hail.annotations.{Annotation, Region}
import is.hail.asm4s.{Code, TypeInfo, Value}
import is.hail.backend.HailStateManager
import is.hail.expr.ir.{Ascending, Descending, EmitCodeBuilder, EmitMethodBuilder, SortOrder}
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.types.physical.stypes.{SCode, SValue}

trait PUnrealizable extends PType {
  private def unsupported: Nothing =
    throw new UnsupportedOperationException(s"$this is not realizable")

  override def byteSize: Long = unsupported

  override def alignment: Long = unsupported

  protected[physical] def _copyFromAddress(
    sm: HailStateManager,
    region: Region,
    srcPType: PType,
    srcAddress: Long,
    deepCopy: Boolean,
  ): Long =
    unsupported

  override def copyFromAddress(
    sm: HailStateManager,
    region: Region,
    srcPType: PType,
    srcAddress: Long,
    deepCopy: Boolean,
  ): Long =
    unsupported

  def unstagedStoreAtAddress(
    sm: HailStateManager,
    addr: Long,
    region: Region,
    srcPType: PType,
    srcAddress: Long,
    deepCopy: Boolean,
  ): Unit =
    unsupported

  override def unstagedStoreJavaObject(sm: HailStateManager, annotation: Annotation, region: Region)
    : Long =
    unsupported

  override def unstagedStoreJavaObjectAtAddress(
    sm: HailStateManager,
    addr: Long,
    annotation: Annotation,
    region: Region,
  ): Unit =
    unsupported

  override def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SValue = unsupported

  override def store(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean)
    : Value[Long] = unsupported

  override def storeAtAddress(
    cb: EmitCodeBuilder,
    addr: Code[Long],
    region: Value[Region],
    value: SValue,
    deepCopy: Boolean,
  ): Unit = unsupported

  override def containsPointers: Boolean =
    throw new UnsupportedOperationException("containsPointers not supported on PUnrealizable")

  override def copiedType: PType = this
}
