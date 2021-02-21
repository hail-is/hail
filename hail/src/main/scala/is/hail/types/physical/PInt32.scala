package is.hail.types.physical

import is.hail.annotations.{Region, UnsafeOrdering, _}
import is.hail.asm4s.{Code, coerce, const, _}
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.stypes.primitives.{SInt32, SInt32Code}
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.types.virtual.TInt32

case object PInt32Optional extends PInt32(false)
case object PInt32Required extends PInt32(true)

class PInt32(override val required: Boolean) extends PNumeric with PPrimitive {
  lazy val virtualType: TInt32.type = TInt32
  def _asIdent = "int32"
  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PInt32")
  override type NType = PInt32

  override def unsafeOrdering(): UnsafeOrdering = new UnsafeOrdering {
    def compare(o1: Long, o2: Long): Int = {
      Integer.compare(Region.loadInt(o1), Region.loadInt(o2))
    }
  }

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering = {
    assert(other isOfType this)
    new CodeOrdering {
      def compareNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Int] =
        Code.invokeStatic2[java.lang.Integer, Int, Int, Int]("compare", x.asInt.intCode(cb), y.asInt.intCode(cb))

      def ltNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asInt.intCode(cb) < y.asInt.intCode(cb)

      def lteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asInt.intCode(cb) <= y.asInt.intCode(cb)

      def gtNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asInt.intCode(cb) > y.asInt.intCode(cb)

      def gteqNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asInt.intCode(cb) >= y.asInt.intCode(cb)

      def equivNonnull(cb: EmitCodeBuilder, x: PCode, y: PCode): Code[Boolean] = x.asInt.intCode(cb).ceq(y.asInt.intCode(cb))
    }
  }

  override def byteSize: Long = 4

  override def zero = coerce[PInt32](const(0))

  override def add(a: Code[_], b: Code[_]): Code[PInt32] = {
    coerce[PInt32](coerce[Int](a) + coerce[Int](b))
  }

  override def multiply(a: Code[_], b: Code[_]): Code[PInt32] = {
    coerce[PInt32](coerce[Int](a) * coerce[Int](b))
  }

  override def sType: SType = SInt32(required)

  def storePrimitiveAtAddress(cb: EmitCodeBuilder, addr: Code[Long], value: SCode): Unit =
    cb.append(Region.storeInt(addr, value.asInt.intCode(cb)))

  override def loadCheapPCode(cb: EmitCodeBuilder, addr: Code[Long]): PCode = new SInt32Code(required, Region.loadInt(addr))

  override def unstagedStoreJavaObjectAtAddress(addr: Long, annotation: Annotation, region: Region): Unit = {
    Region.storeInt(addr, annotation.asInstanceOf[Int])
  }
}

object PInt32 {
  def apply(required: Boolean = false) = if (required) PInt32Required else PInt32Optional

  def unapply(t: PInt32): Option[Boolean] = Option(t.required)
}
