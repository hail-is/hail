package is.hail.types.physical

import is.hail.annotations.{Region, UnsafeOrdering, _}
import is.hail.asm4s.Code
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.primitives.{SBoolean, SBooleanCode, SBooleanValue}
import is.hail.types.virtual.TBoolean
import is.hail.utils.toRichBoolean

case object PBooleanOptional extends PBoolean(false)
case object PBooleanRequired extends PBoolean(true)

class PBoolean(override val required: Boolean) extends PType with PPrimitive {
  lazy val virtualType: TBoolean.type  = TBoolean

  def _asIdent = "bool"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = sb.append("PBoolean")

  override def unsafeOrdering(): UnsafeOrdering = new UnsafeOrdering {
    def compare(o1: Long, o2: Long): Int = {
      java.lang.Boolean.compare(Region.loadBoolean(o1), Region.loadBoolean(o2))
    }
  }

  override def byteSize: Long = 1

  def sType: SBoolean.type = SBoolean

  def storePrimitiveAtAddress(cb: EmitCodeBuilder, addr: Code[Long], value: SValue): Unit = {
    cb += Region.storeBoolean(addr, value.asBoolean.boolCode(cb))
  }

  override def loadCheapSCode(cb: EmitCodeBuilder, addr: Code[Long]): SBooleanValue =
  new SBooleanCode(Region.loadBoolean(addr)).memoize(cb, "loadCheapSCodeField")

  override def loadCheapSCodeField(cb: EmitCodeBuilder, addr: Code[Long]): SBooleanValue =
    new SBooleanCode(Region.loadBoolean(addr)).memoizeField(cb, "loadCheapSCodeField")

  override def unstagedStoreJavaObjectAtAddress(addr: Long, annotation: Annotation, region: Region): Unit = {
    Region.storeByte(addr, annotation.asInstanceOf[Boolean].toByte)
  }
}

object PBoolean {
  def apply(required: Boolean = false): PBoolean = if (required) PBooleanRequired else PBooleanOptional

  def unapply(t: PBoolean): Option[Boolean] = Option(t.required)
}
