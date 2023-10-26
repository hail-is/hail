package is.hail.types.physical

import is.hail.annotations.{Annotation, Region, UnsafeOrdering}
import is.hail.asm4s._
import is.hail.backend.HailStateManager
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.types.virtual.Type
import is.hail.utils._


object StoredCodeTuple {
  def canStore(ti: TypeInfo[_]): Boolean = ti match {
    case IntInfo => true
    case LongInfo => true
    case FloatInfo => true
    case DoubleInfo => true
    case BooleanInfo => true
    case _ => false
  }

  def byteSize(ti: TypeInfo[_]): Long = ti match {
    case IntInfo => 4
    case LongInfo => 8
    case FloatInfo => 4
    case DoubleInfo => 8
    case BooleanInfo => 1
  }
}

class StoredCodeTuple(tis: Array[TypeInfo[_]]) {
  tis.foreach(ti => require(StoredCodeTuple.canStore(ti)))

  private[this] val tiByteSize = tis.map(StoredCodeTuple.byteSize)
  private[this] val fieldOffsets = new Array[Long](tis.length)
  val byteSize: Long = getByteSizeAndOffsets(tiByteSize, tiByteSize, 0, fieldOffsets)
  val alignment: Long = tiByteSize.max

  def store(cb: EmitCodeBuilder, addr: Value[Long], codes: IndexedSeq[Code[_]]): Unit = {
    assert(codes.length == tis.length)
    tis.indices.foreach { i =>
      val ti = tis(i)
      val c = codes(i)
      assert(c.ti == ti)

      val offset = addr + fieldOffsets(i)

      ti match {
        case IntInfo => cb += Region.storeInt(offset, coerce[Int](c))
        case LongInfo => cb += Region.storeLong(offset, coerce[Long](c))
        case FloatInfo => cb += Region.storeFloat(offset, coerce[Float](c))
        case DoubleInfo => cb += Region.storeDouble(offset, coerce[Double](c))
        case BooleanInfo => cb += Region.storeBoolean(offset, coerce[Boolean](c))
      }
    }
  }

  def load(cb: EmitCodeBuilder, addr: Value[Long]): IndexedSeq[Code[_]] = {
    tis.indices.map { i =>
      val ti = tis(i)
      val offset = addr + fieldOffsets(i)
      ti match {
        case IntInfo => Region.loadInt(offset)
        case LongInfo => Region.loadLong(offset)
        case FloatInfo => Region.loadFloat(offset)
        case DoubleInfo => Region.loadDouble(offset)
        case BooleanInfo => Region.loadBoolean(offset)
      }
    }
  }

  def loadValues(cb: EmitCodeBuilder, addr: Value[Long]): IndexedSeq[Value[_]] = {
    tis.indices.map { i =>
      val ti = tis(i)
      val offset = addr + fieldOffsets(i)
      val code = ti match {
        case IntInfo => Region.loadInt(offset)
        case LongInfo => Region.loadLong(offset)
        case FloatInfo => Region.loadFloat(offset)
        case DoubleInfo => Region.loadDouble(offset)
        case BooleanInfo => Region.loadBoolean(offset)
      }
      cb.memoizeAny(code, ti)
    }
  }
}

case class StoredSTypePType(sType: SType, required: Boolean) extends PType {

  private[this] lazy val ct = new StoredCodeTuple(sType.settableTupleTypes().toArray)

  override def virtualType: Type = sType.virtualType

  override def store(cb: EmitCodeBuilder, region: Value[Region], value: SValue, deepCopy: Boolean): Value[Long] = {
    val addr = cb.memoize(region.allocate(ct.alignment, ct.byteSize))
    ct.store(cb, addr, value.st.coerceOrCopy(cb, region, value, deepCopy).valueTuple.map(_.get))
    addr
  }

  override def storeAtAddress(cb: EmitCodeBuilder, addr: Code[Long], region: Value[Region], value: SValue, deepCopy: Boolean): Unit = {
    ct.store(cb, cb.newLocal[Long]("stored_stype_ptype_addr", addr), value.st.coerceOrCopy(cb, region, value, deepCopy).valueTuple.map(_.get))
  }

  override def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SValue = {
    sType.fromValues(ct.loadValues(cb, cb.newLocal[Long]("stored_stype_ptype_loaded_addr")))
  }

  override def loadFromNested(addr: Code[Long]): Code[Long] = addr

  override def deepRename(t: Type): PType = StoredSTypePType(sType.castRename(t), required)

  def byteSize: Long = ct.byteSize

  override def alignment: Long = ct.alignment

  override def containsPointers: Boolean = sType.containsPointers

  override def setRequired(required: Boolean): PType = if (required == this.required) this else StoredSTypePType(sType, required)

  def unsupportedCanonicalMethod: Nothing = throw new UnsupportedOperationException("not supported on StoredStypePType")

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append(sType.toString)

  override def unsafeOrdering(sm: HailStateManager, rightType: PType): UnsafeOrdering = unsupportedCanonicalMethod

  override def unsafeOrdering(sm: HailStateManager): UnsafeOrdering = unsupportedCanonicalMethod

  def unstagedLoadFromNested(addr: Long): Long = unsupportedCanonicalMethod

  def unstagedStoreJavaObject(sm: HailStateManager, annotation: Annotation, region: Region): Long = unsupportedCanonicalMethod

  def unstagedStoreJavaObjectAtAddress(sm: HailStateManager, addr: Long, annotation: Annotation, region: Region): Unit = unsupportedCanonicalMethod

  override def _copyFromAddress(sm: HailStateManager, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = unsupportedCanonicalMethod

  override def unstagedStoreAtAddress(sm: HailStateManager, addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = unsupportedCanonicalMethod

  override def _asIdent: String = "stored_stype_ptype"

  override def copiedType: PType = {
    val copiedSType = sType.copiedType
    if (copiedSType.eq(sType))
      this
    else
      StoredSTypePType(copiedSType, required)
  }
}
