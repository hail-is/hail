package is.hail.expr.types.physical

import is.hail.annotations.{UnsafeUtils, _}
import is.hail.asm4s._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TArray
import org.json4s.jackson.JsonMethods

import scala.reflect.{ClassTag, _}

final case class PArray(elementType: PType, override val required: Boolean = false) extends PContainer with PStreamable {
  lazy val virtualType: TArray = TArray(elementType.virtualType, required)

  val elementByteSize: Long = UnsafeUtils.arrayElementSize(elementType)

  val contentsAlignment: Long = elementType.alignment.max(4)

  override def pyString(sb: StringBuilder): Unit = {
    sb.append("array<")
    elementType.pyString(sb)
    sb.append('>')
  }
  override val fundamentalType: PArray = {
    if (elementType == elementType.fundamentalType)
      this
    else
      this.copy(elementType = elementType.fundamentalType)
  }

  def _toPretty = s"Array[$elementType]"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("Array[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = {
    assert(this isOfType other)
    CodeOrdering.iterableOrdering(this, other.asInstanceOf[PArray], mb)
  }

  def checkedConvertFrom(mb: EmitMethodBuilder, r: Code[Region], value: Code[_], pt: PType, msg: String): Code[Long] = {
    val pta = pt.asInstanceOf[PArray]
    assert(pta.elementType.isPrimitive)
    val vOff = coerce[Long](value)
    val len = pta.loadLength(vOff)
    if (pta.elementType.required == elementType.required)
      coerce[Long](value)
    else {
      if (pta.elementType.required) {
        // convert from required to non-required
        val offset = mb.newField[Long]
        Code(
          offset := allocate(r, len),
          stagedInitialize(offset, len),
          Region.copyFrom(pta.elementsOffset(len), offset, len.toL * elementByteSize),
          offset
        )
      } else {
        //  convert from non-required to required
        val offset = mb.newField[Long]
        val i = mb.newField[Int]
        Code(
          offset := allocate(r, len),
          stagedInitialize(offset, len),
          i := 0,
          Code.whileLoop(i < len,
            pta.isElementMissing(vOff, i).orEmpty(Code._fatal(s"${msg}: convertFrom $pt failed: element missing.")),
            i := i + 1
          ),
          Region.copyFrom(pta.elementsOffset(len), offset, len.toL * elementByteSize),
          offset
        )
      }
    }
  }
}
