package is.hail.types.physical

import is.hail.annotations.{Annotation, Region, UnsafeOrdering}
import is.hail.asm4s.{Code, TypeInfo, Value}
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.types.virtual.Type
import is.hail.utils._

class StoredCodeTuple(tis: IndexedSeq[TypeInfo[_]]) {
  val byteSize: Long = 0L // unimplemented
  val alignment: Long = 0L // unimplemented

  def store(cb: EmitCodeBuilder, addr: Value[Long], codes: IndexedSeq[Code[_]]): Unit = ???

  def load(cb: EmitCodeBuilder, addr: Value[Long]): IndexedSeq[Code[_]] = ???
}

case class StoredSTypePType(sType: SType, required: Boolean) extends PType {

  private[this] lazy val ct = new StoredCodeTuple(sType.codeTupleTypes())

  override def virtualType: Type = sType.virtualType

  override def store(cb: EmitCodeBuilder, region: Value[Region], value: SCode, deepCopy: Boolean): Code[Long] = {
    val addr = cb.newLocal[Long]("stored_stype_ptype_addr", region.allocate(ct.alignment, ct.byteSize))
    ct.store(cb, addr, value.st.coerceOrCopy(cb, region, value, deepCopy).makeCodeTuple(cb))
    addr
  }

  override def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SCode, deepCopy: Boolean): Unit = {
    ct.store(cb, cb.newLocal[Long]("stored_stype_ptype_addr", addr), value.st.coerceOrCopy(cb, region, value, deepCopy).makeCodeTuple(cb))
  }

  override def loadCheapPCode(cb: EmitCodeBuilder, addr: Code[Long]): SCode = {
    sType.fromCodes(ct.load(cb, cb.newLocal[Long]("stored_stype_ptype_loaded_addr")))
  }

  override def loadFromNested(addr: Code[Long]): Code[Long] = addr

  override def deepRename(t: Type): PType = StoredSTypePType(sType.castRename(t), required)

  def byteSize: Long = ct.byteSize
  override def alignment: Long = ct.alignment

  override def containsPointers: Boolean = ??? // need sType.containsPointers

  override def setRequired(required: Boolean): PType = ???

  def unsupportedCanonicalMethod: Nothing = ???

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = ???

  override def unsafeOrdering(rightType: PType): UnsafeOrdering = unsupportedCanonicalMethod

  override def unsafeOrdering(): UnsafeOrdering = unsupportedCanonicalMethod

  override def deepInnerRequired(required: Boolean): PType = unsupportedCanonicalMethod

  def unstagedLoadFromNested(addr: Long): Long = unsupportedCanonicalMethod

  def unstagedStoreJavaObject(annotation: Annotation, region: Region): Long = unsupportedCanonicalMethod

  def unstagedStoreJavaObjectAtAddress(addr: Long, annotation: Annotation, region: Region): Unit = unsupportedCanonicalMethod

  def unstagedStoreAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = unsupportedCanonicalMethod

  override def copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = unsupportedCanonicalMethod

  override def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = unsupportedCanonicalMethod

  override def _asIdent: String = ???
}
