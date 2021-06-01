package is.hail.types.encoded
import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.stypes.concrete.{SIndexablePointerSettable, SNestedArray, SNestedArraySettable}
import is.hail.types.physical.stypes.interfaces.SContainer
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.types.virtual.{TArray, Type}
import is.hail.utils._

class EFlattenedArray(override val required: Boolean, nestedRequiredness: Array[Boolean], val innerType: EContainer) extends EType {
  def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit = v match {
    case v: SNestedArraySettable =>
      require(nestedRequiredness.length == v.st.levels)
      val i = cb.newLocal[Int]("i", 0)
      for (j <- 0 until v.st.levels) {
        val length = v.lengths(j)
        val missing = v.missing(j)
        val offsets = v.offsets(j)
        cb += out.writeInt(length)
        if (missing != null) {
          cb += out.writeBytes(missing, UnsafeUtils.packBitsToBytes(length))
        }
        cb.forLoop(cb.assign(i, 0), i < length, cb.assign(i, i + 1), {
          cb.ifx(!Region.loadBit(missing, i.toL), {
            val len = Region.loadInt(offsets + (i + const(1)).toL * 4L) - Region.loadInt(offsets + i.toL * 4L)
            cb += out.writeInt(len)
          })
        })
      }
      val innerEncode = innerType.buildEncoder(v.st.baseContainerType, cb.emb.ecb)
      innerEncode(cb, v.values, out)
    case v: SIndexablePointerSettable =>
      ???
  }

  def _buildDecoder(cb: EmitCodeBuilder, t: Type, region: Value[Region], in: Value[InputBuffer]): SCode = ???

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit = {
    val len = cb.newLocal[Int]("len")
    val maxLen = cb.newLocal[Int]("maxLen")
    val missing = cb.newLocal[Long]("missing")
    val i = cb.newLocal[Int]("i")
    for (req <- nestedRequiredness) {
      cb.assign(len, in.readInt())
      cb.ifx(!const(req) && len.cne(0) && (len > maxLen || maxLen.ceq(0)), {
        cb.assign(maxLen, len)
        cb.assign(missing, r.allocate(1, UnsafeUtils.packBitsToBytes(maxLen)))
      })
      if (!req) {
        cb += in.readBytes(r, missing, UnsafeUtils.packBitsToBytes(len))
      }
      cb.forLoop(cb.assign(i, 0), i < len, cb.assign(i, i + 1), {
        if (req) {
          cb += in.skipInt()
        } else {
          cb.ifx(!Region.loadBit(missing, i.toL), cb += in.skipInt())
        }
      })
    }
  }

  def _asIdent: String = ???

  def _toPretty: String = ???

  def _decodedSType(requestedType: Type): SType = {
    def go(level: Int, ty: Type): (Int, SContainer) = ty match {
      case TArray(a: TArray) => go(level + 1, a)
      case TArray(_) => (level, innerType.decodedSType(ty).asInstanceOf)
    }
    val (levels, base) = go(0, requestedType)
    SNestedArray(levels, base)
  }

  override def setRequired(required: Boolean): EFlattenedArray = if (required == this.required) this else new EFlattenedArray(required, nestedRequiredness, innerType)
}
