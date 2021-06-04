package is.hail.types.encoded
import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.stypes.concrete.{SNestedArray, SNestedArrayCode}
import is.hail.types.physical.stypes.interfaces.SContainer
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.types.virtual.{TArray, Type}
import is.hail.utils._

import scala.annotation.tailrec

case class EFlattenedArray(override val required: Boolean, nestedRequiredness: IndexedSeq[Boolean], innerType: EContainer) extends EType {
  def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit = ???

  def _buildDecoder(cb: EmitCodeBuilder, t: Type, region: Value[Region], in: Value[InputBuffer]): SCode = {
    val st = decodedSType(t).asInstanceOf[SNestedArray]
    if (nestedRequiredness.isEmpty) {
      val sc = innerType._buildDecoder(cb, t, region, in).asIndexable
      new SNestedArrayCode(st, const(0), sc.loadLength(), FastIndexedSeq(), FastIndexedSeq(), sc)
    } else {
      val i = cb.newLocal[Int]("i")
      val len = cb.newLocal[Int]("len")
      val end = cb.newLocal[Int]("end")
      val missing = nestedRequiredness.zipWithIndex.filter(_._1).map { case (_, i) => cb.newLocal[Long](s"nested_array_decode_missing_$i") }
      val offsets = Array.tabulate(nestedRequiredness.length)(i => cb.newLocal[Long](s"nested_array_decode_offset_$i"))
      val cur = cb.newLocal[Int]("cur")
      var first = true
      var mj = 0
      for ((r, j) <- nestedRequiredness.zipWithIndex) {
        cb.assign(len, in.readInt())
        if (first) {
          cb.assign(end, len)
          first = false
        }

        if (!r) {
          cb.assign(missing(mj), region.allocate(1L, UnsafeUtils.packBitsToBytes(len)))
          cb += in.readBytes(region, missing(mj), UnsafeUtils.packBitsToBytes(len))
        }
        cb.assign(offsets(j), region.allocate(4L, 4L * (len + 1).toL))
        cb.assign(cur, 0)
        cb += Region.storeInt(offsets(j), cur)
        cb.forLoop(cb.assign(i, 1), i <= len, cb.assign(i, i + 1), {
          if (r) {
            cb.assign(cur, cur + in.readInt())
          } else {
            cb.ifx(!Region.loadBit(missing(mj), (i - 1).toL), {
              cb.assign(cur, cur + in.readInt())
            })
          }
          cb += Region.storeInt(offsets(j) + i.toL * 4L, cur)
        })
        mj += (!r).toInt
      }

      val values = innerType.buildDecoder(st.baseContainerType.virtualType, cb.emb.ecb)(cb, region, in).asIndexable

      new SNestedArrayCode(st, const(0), end, missing.map(_.get), offsets.map(_.get), values)
    }
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit = {
    val len = cb.newLocal[Int]("len")
    val maxLen = cb.newLocal[Int]("maxLen")
    val missing = cb.newLocal[Long]("missing")
    val i = cb.newLocal[Int]("i")
    for (req <- nestedRequiredness) {
      cb.assign(len, in.readInt())
      cb.ifx(!const(req) && len.cne(0) && (len > maxLen || maxLen.ceq(0)), {
        cb.assign(maxLen, len)
        cb.assign(missing, r.allocate(const(1L), UnsafeUtils.packBitsToBytes(maxLen).toL))
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

  def _asIdent: String = s"flat_array_w_${ nestedRequiredness.map(if (_) "r" else "o").mkString }_of_${innerType.asIdent}"

  def _toPretty: String = s"EFlattenedArray[${ nestedRequiredness.map(if (_) "r" else "o").mkString }, $innerType]"

  def _decodedSType(requestedType: Type): SType = {
    @tailrec def go(ty: Type): SContainer = ty match {
      case TArray(a: TArray) => go(a)
      case TArray(_) => innerType.decodedSType(ty).asInstanceOf
    }
    val base = go(requestedType)
    SNestedArray(nestedRequiredness, base)
  }

  override def setRequired(newRequired: Boolean): EFlattenedArray = EFlattenedArray(newRequired, nestedRequiredness, innerType)
}
