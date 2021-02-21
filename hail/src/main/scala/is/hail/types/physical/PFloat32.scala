package is.hail.types.physical

import is.hail.annotations._
import is.hail.asm4s.{Code, _}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.stypes.primitives.{SFloat32, SFloat32Code}
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.types.virtual.TFloat32

case object PFloat32Optional extends PFloat32(false)
case object PFloat32Required extends PFloat32(true)

class PFloat32(override val required: Boolean) extends PNumeric with PPrimitive {
  lazy val virtualType: TFloat32.type = TFloat32

  override type NType = PFloat32

  def _asIdent = "float32"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PFloat32")

  override def unsafeOrdering(): UnsafeOrdering = new UnsafeOrdering {
    def compare(o1: Long, o2: Long): Int = {
      java.lang.Float.compare(Region.loadFloat(o1), Region.loadFloat(o2))
    }
  }

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering = {
    assert(other isOfType this)
    new CodeOrdering {
      def compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] =
        Code.invokeStatic2[java.lang.Float, Float, Float, Int]("compare", x.asFloat.floatCode(cb), y.asFloat.floatCode(cb))

      def ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asFloat.floatCode(cb) < y.asFloat.floatCode(cb)

      def lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asFloat.floatCode(cb) <= y.asFloat.floatCode(cb)

      def gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asFloat.floatCode(cb) > y.asFloat.floatCode(cb)

      def gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asFloat.floatCode(cb) >= y.asFloat.floatCode(cb)

      def equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asFloat.floatCode(cb).ceq(y.asFloat.floatCode(cb))
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

  override def sType: SType = SFloat32(required)

  def storePrimitiveAtAddress(cb: EmitCodeBuilder, addr: Code[Long], value: SCode): Unit =
    cb.append(Region.storeFloat(addr, value.asFloat.floatCode(cb)))

  override def loadCheapPCode(cb: EmitCodeBuilder, addr: Code[Long]): PCode = new SFloat32Code(required, Region.loadFloat(addr))

  override def unstagedStoreJavaObjectAtAddress(addr: Long, annotation: Annotation, region: Region): Unit = {
    Region.storeFloat(addr, annotation.asInstanceOf[Float])
  }
}

object PFloat32 {
  def apply(required: Boolean = false): PFloat32 = if (required) PFloat32Required else PFloat32Optional

  def unapply(t: PFloat32): Option[Boolean] = Option(t.required)
}
