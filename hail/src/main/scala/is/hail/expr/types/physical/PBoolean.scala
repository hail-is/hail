package is.hail.expr.types.physical

import is.hail.annotations.{Region, UnsafeOrdering, _}
import is.hail.asm4s.{Code, MethodBuilder}
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TBoolean

case object PBooleanOptional extends PBoolean(false)
case object PBooleanRequired extends PBoolean(true)

class PBoolean(override val required: Boolean) extends PType {
  lazy val virtualType: TBoolean = TBoolean(required)

  def _asIdent = "bool"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PBoolean")

  override def unsafeOrdering(): UnsafeOrdering = new UnsafeOrdering {
    def compare(r1: Region, o1: Long, r2: Region, o2: Long): Int = {
      java.lang.Boolean.compare(Region.loadBoolean(o1), Region.loadBoolean(o2))
    }
  }

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(other isOfType this)
    new CodeOrderingCompareConsistentWithOthers {
      type T = Boolean

      def compareNonnull(x: Code[T], y: Code[T]): Code[Int] =
        Code.invokeStatic[java.lang.Boolean, Boolean, Boolean, Int]("compare", x, y)
    }
  }

  override def byteSize: Long = 1

  def storeShallowAtOffset(dstAddress: Code[Long], srcAddress: Code[Long]): Code[Unit] =
    Region.storeBoolean(dstAddress, Region.loadBoolean(srcAddress))

  def storeShallowAtOffset(dstAddress: Long, srcAddress: Long) =
    Region.storeBoolean(dstAddress, Region.loadBoolean(srcAddress))

  def copyFromType(region: Region, srcPType: PType, srcAddress: Long, forceDeep: Boolean): Long =
    srcAddress

  def copyFromType(mb: MethodBuilder, region: Code[Region], srcPType: PType, srcAddress: Code[Long], forceDeep: Boolean): Code[Long] =
    srcAddress

  def copyFromTypeAndStackValue(mb: MethodBuilder, region: Code[Region], srcPType: PType, stackValue: Code[_], forceDeep: Boolean): Code[_] =
    stackValue
}

object PBoolean {
  def apply(required: Boolean = false): PBoolean = if (required) PBooleanRequired else PBooleanOptional

  def unapply(t: PBoolean): Option[Boolean] = Option(t.required)
}
