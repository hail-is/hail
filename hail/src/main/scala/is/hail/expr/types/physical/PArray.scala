package is.hail.expr.types.physical

import is.hail.annotations.{UnsafeUtils, _}
import is.hail.asm4s._
import is.hail.check.Gen
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TArray
import is.hail.utils._
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

  def checkedConvertFrom(mb: EmitMethodBuilder, r: Code[Region], value: Code[_], otherPT: PType, msg: String): Code[Long] = {
    val otherPTA = otherPT.asInstanceOf[PArray]
    assert(otherPTA.elementType.isPrimitive)
    val oldOffset = coerce[Long](value)
    val len = otherPTA.loadLength(oldOffset)
    if (otherPTA.elementType.required == elementType.required) {
      coerce[Long](value)
    }
    else {
      if (otherPTA.elementType.required) {
        // convert from required to non-required
        val newOffset = mb.newField[Long]
        Code(
          newOffset := allocate(r, len),
          stagedInitialize(newOffset, len),
          Region.copyFrom(otherPTA.elementsOffset(len), newOffset, len.toL * elementByteSize),
          newOffset
        )
      } else {
        //  convert from non-required to required
        val newOffset = mb.newField[Long]
        val i = mb.newField[Int]
        Code(
          newOffset := allocate(r, len),
          stagedInitialize(newOffset, len),
          i := 0,
          Code.whileLoop(i < len,
            otherPTA.isElementMissing(oldOffset, i).orEmpty(Code._fatal(s"${msg}: convertFrom $otherPT failed: element missing.")),
            i := i + 1
          ),
          Region.storeInt(newOffset.load(), len),
          Region.copyFrom(oldOffset + otherPTA.elementsOffset(len), newOffset + 4L, len.toL * elementByteSize),
          newOffset
        )
      }
    }
  }
}
