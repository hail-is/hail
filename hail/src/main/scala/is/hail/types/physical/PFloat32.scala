package is.hail.types.physical

import is.hail.annotations._
import is.hail.asm4s.{Code, _}
import is.hail.backend.HailStateManager
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.primitives.{SFloat32, SFloat32Value}
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.virtual.TFloat32

case object PFloat32Optional extends PFloat32(false)
case object PFloat32Required extends PFloat32(true)

class PFloat32(override val required: Boolean) extends PNumeric with PPrimitive {
  lazy val virtualType: TFloat32.type = TFloat32

  override type NType = PFloat32

  def _asIdent = "float32"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PFloat32")

  override def unsafeOrdering(sm: HailStateManager): UnsafeOrdering = new UnsafeOrdering {
    def compare(o1: Long, o2: Long): Int = {
      java.lang.Float.compare(Region.loadFloat(o1), Region.loadFloat(o2))
    }
  }

  override def byteSize: Long = 4

  override def zero = coerce[PFloat32](const(0.0f))

  override def add(a: Code[_], b: Code[_]): Code[PFloat32] = {
    coerce[PFloat32](coerce[Float](a) + coerce[Float](b))
  }

  override def multiply(a: Code[_], b: Code[_]): Code[PFloat32] = {
    coerce[PFloat32](coerce[Float](a) * coerce[Float](b))
  }

  override def sType: SType = SFloat32

  def storePrimitiveAtAddress(cb: EmitCodeBuilder, addr: Code[Long], value: SValue): Unit =
    cb.append(Region.storeFloat(addr, value.asFloat.value))

  override def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SFloat32Value =
    new SFloat32Value(cb.memoize(Region.loadFloat(addr)))

  override def unstagedStoreJavaObjectAtAddress(sm: HailStateManager, addr: Long, annotation: Annotation, region: Region): Unit = {
    Region.storeFloat(addr, annotation.asInstanceOf[Float])
  }
}

object PFloat32 {
  def apply(required: Boolean = false): PFloat32 = if (required) PFloat32Required else PFloat32Optional

  def unapply(t: PFloat32): Option[Boolean] = Option(t.required)
}
