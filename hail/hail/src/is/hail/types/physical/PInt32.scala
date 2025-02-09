package is.hail.types.physical

import is.hail.annotations._
import is.hail.asm4s.{coerce, const, Code, _}
import is.hail.backend.HailStateManager
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.primitives.{SInt32, SInt32Value}
import is.hail.types.virtual.TInt32

case object PInt32Optional extends PInt32(false)
case object PInt32Required extends PInt32(true)

class PInt32(override val required: Boolean) extends PNumeric with PPrimitive {
  lazy val virtualType: TInt32.type = TInt32
  def _asIdent = "int32"
  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PInt32")
  override type NType = PInt32

  override def unsafeOrdering(sm: HailStateManager): UnsafeOrdering = new UnsafeOrdering {
    def compare(o1: Long, o2: Long): Int =
      Integer.compare(Region.loadInt(o1), Region.loadInt(o2))
  }

  override def byteSize: Long = 4

  override def zero = coerce[PInt32](const(0))

  override def add(a: Code[_], b: Code[_]): Code[PInt32] =
    coerce[PInt32](coerce[Int](a) + coerce[Int](b))

  override def multiply(a: Code[_], b: Code[_]): Code[PInt32] =
    coerce[PInt32](coerce[Int](a) * coerce[Int](b))

  override def sType: SType = SInt32

  def storePrimitiveAtAddress(cb: EmitCodeBuilder, addr: Code[Long], value: SValue): Unit =
    cb.append(Region.storeInt(addr, value.asInt.value))

  override def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SInt32Value =
    new SInt32Value(cb.memoize(Region.loadInt(addr)))

  override def unstagedStoreJavaObjectAtAddress(
    sm: HailStateManager,
    addr: Long,
    annotation: Annotation,
    region: Region,
  ): Unit =
    Region.storeInt(addr, annotation.asInstanceOf[Int])

  def unstagedLoadFromAddress(addr: Long): Int = Region.loadInt(addr)
}

object PInt32 {
  def apply(required: Boolean = false) = if (required) PInt32Required else PInt32Optional

  def unapply(t: PInt32): Option[Boolean] = Option(t.required)
}
