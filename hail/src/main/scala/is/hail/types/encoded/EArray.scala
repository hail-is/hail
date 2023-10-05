package is.hail.types.encoded

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical._
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SIndexablePointerValue}
import is.hail.types.physical.stypes.interfaces.SIndexableValue
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.virtual._
import is.hail.utils._

final case class EArray(val elementType: EType, override val required: Boolean = false) extends EContainer {
  def _decodedSType(requestedType: Type): SType = {
    val elementPType = elementType.decodedPType(requestedType.asInstanceOf[TContainer].elementType)
    requestedType match {
      case _: TSet =>
        SIndexablePointer(PCanonicalSet(elementPType, false))
      case _: TArray =>
        SIndexablePointer(PCanonicalArray(elementPType, false))
      case _: TDict =>
        val et = elementPType.asInstanceOf[PStruct]
        SIndexablePointer(PCanonicalDict(et.fieldType("key"), et.fieldType("value"), false))
    }
  }

  def buildPrefixEncoder(cb: EmitCodeBuilder, value: SIndexableValue,
    out: Value[OutputBuffer], prefixLength: Code[Int]
  ): Unit = {
    val prefixLen = cb.newLocal[Int]("prefixLen", prefixLength)
    val i = cb.newLocal[Int]("i", 0)

    cb += out.writeInt(prefixLen)

    value.st match {
      case s@SIndexablePointer(_: PCanonicalArray | _: PCanonicalSet | _:PCanonicalDict)
        if s.pType.elementType.required == elementType.required =>
        val pArray = s.pType match {
          case t: PCanonicalArray => t
          case t: PCanonicalSet => t.arrayRep
          case t: PCanonicalDict => t.arrayRep
        }

        val array = value.asInstanceOf[SIndexablePointerValue].a
        if (!elementType.required) {
          val nMissingLocal = cb.newLocal[Int]("nMissingBytes", pArray.nMissingBytes(prefixLen))
          cb.ifx(nMissingLocal > 0, {
            cb += out.writeBytes(array + const(pArray.lengthHeaderBytes), nMissingLocal - 1)
            cb += out.writeByte((Region.loadByte(array + const(pArray.lengthHeaderBytes)
              + (nMissingLocal - 1).toL) & EType.lowBitMask(prefixLen)).toB)
          })
        }
      case _ =>
        if (elementType.required) {
          cb.ifx(value.hasMissingValues(cb), cb._fatal("cannot encode indexable with missing element(s) to required EArray!"))
        } else {
          val b = Code.newLocal[Int]("b")
          val shift = Code.newLocal[Int]("shift")
          cb.assign(b, 0)
          cb.assign(shift, 0)
          cb.whileLoop(i < prefixLen, {
            cb.ifx(value.isElementMissing(cb, i), cb.assign(b, b | (const(1) << shift)))
            cb.assign(shift, shift + 1)
            cb.assign(i, i + 1)
            cb.ifx(shift.ceq(8), {
              cb.assign(shift, 0)
              cb += out.writeByte(b.toB)
              cb.assign(b, 0)
            })
          })
          cb.ifx(shift > 0, cb += out.writeByte(b.toB))
        }
    }

    cb.forLoop(cb.assign(i, 0), i < prefixLen, cb.assign(i, i + 1), {
      value.loadElement(cb, i).consume(cb, {
        if (elementType.required)
          cb._fatal(s"required array element saw missing value at index ", i.toS, " in encode")
      }, { pc =>
        elementType.buildEncoder(pc.st, cb.emb.ecb)
          .apply(cb, pc, out)
      })
    })
  }

  override def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit = {
    val ind = v.asInstanceOf[SIndexableValue]
    buildPrefixEncoder(cb, ind, out, ind.loadLength())
  }

  override def _buildDecoder(cb: EmitCodeBuilder, t: Type, region: Value[Region], in: Value[InputBuffer]): SValue = {
    val st = decodedSType(t).asInstanceOf[SIndexablePointer]

    val arrayType: PCanonicalArray = st.pType match {
      case t: PCanonicalArray => t
      case t: PCanonicalSet => t.arrayRep
      case t: PCanonicalDict => t.arrayRep
    }

    val len = cb.newLocal[Int]("len", in.readInt())
    val array = cb.newLocal[Long]("array", arrayType.allocate(region, len))
    arrayType.storeLength(cb, array, len)

    val readElemF = elementType.buildInplaceDecoder(arrayType.elementType, cb.emb.ecb)
    val pastLastFull64BlockOff = cb.memoize(arrayType.elementOffset(array, len, len - 64))
    val pastLastOff = cb.memoize(arrayType.elementOffset(array, len, len))

    if (elementType.required) {
      // val i = cb.newLocal[Int]("i")
      // cb.forLoop(cb.assign(i, 0), i < len, cb.assign(i, i + 1), {
      //   val elemOff = cb.memoize(arrayType.elementOffset(array, len, i))
      //   readElemF(cb, region, elemOff, in)
      // })
      //
      // cb.forLoop({}, elemOff < pastLastFull64BlockOff, {}, {
      //   for (_ <- 0 to 63) {
      //     readElemF(cb, region, elemOff, in)
      //     cb.assign(elemOff, arrayType.nextElementOffset(elemOff))
      //   }
      // })
      val elemOff = cb.newLocal[Long]("elemOff", arrayType.firstElementOffset(array, len))
      cb.forLoop({}, elemOff < pastLastOff, cb.assign(elemOff, arrayType.nextElementOffset(elemOff)), {
        readElemF(cb, region, elemOff, in)
      })
    } else {
      cb += in.readBytes(region, array + const(arrayType.lengthHeaderBytes), arrayType.nMissingBytes(len))

      def unsetRightMostBit(x: Value[Long]): Code[Long] =
        x & (x - 1)

      val presentBits = cb.newLocal[Long]("presentBits", 0L)
      val mbyteOffset = cb.newLocal[Long]("mbyteOffset", array + arrayType.lengthHeaderBytes)
      val blockOff = cb.newLocal[Long]("blockOff", arrayType.firstElementOffset(array, len))
      val inBlockIndexToPresentValue = cb.newLocal[Int]("inBlockIndexToPresentValue", 0)

      cb.forLoop(
        {},
        blockOff < pastLastFull64BlockOff,
        cb.assign(blockOff, arrayType.incrementElementOffset(blockOff, 64)),
        {
          cb.assign(presentBits, ~Region.loadLong(mbyteOffset).reverseBytes)
          cb.assign(mbyteOffset, mbyteOffset + 8)
          cb.whileLoop(presentBits.cne(0L), {
            cb.assign(inBlockIndexToPresentValue, presentBits.numberOfTrailingZeros)
            val elemOff = cb.memoize(
              arrayType.incrementElementOffset(blockOff, inBlockIndexToPresentValue))
            readElemF(cb, region, elemOff, in)
            cb.assign(presentBits, unsetRightMostBit(presentBits))
          })
        }
      )
      val i = cb.newLocal[Int]("i", len - (len % 64))
      cb.forLoop({}, i < len, cb.assign(i, i + 1), {
        cb.ifx(arrayType.isElementDefined(array, i),
          readElemF(cb, region, cb.memoize(arrayType.elementOffset(array, len, i)), in))
      })
      // cb.forLoop({}, blockOff < pastLastOff, {}, {
      //   cb.assign(presentBits, (~(Region.loadByte(mbyteOffset).toL)) & 0xFF.toByte)
      //   cb.assign(mbyteOffset, mbyteOffset + 1)
      //   cb.whileLoop(presentBits.cne(0L), {
      //     cb.assign(inBlockIndexToPresentValue, presentBits.numberOfTrailingZeros)
      //     val elemOff = cb.memoize(
      //       arrayType.incrementElementOffset(blockOff, inBlockIndexToPresentValue))
      //     readElemF(cb, region, elemOff, in)
      //     cb.assign(presentBits, unsetRightMostBit(presentBits))
      //   })
      //   cb.assign(blockOff, arrayType.incrementElementOffset(blockOff, 8))
      // })
    }

    new SIndexablePointerValue(st, array, len, cb.memoize(arrayType.firstElementOffset(array, len)))
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit = {
    val skip = elementType.buildSkip(cb.emb.ecb)
    val len = cb.newLocal[Int]("len", in.readInt())
    val i = cb.newLocal[Int]("i")
    if (elementType.required) {
      cb.forLoop(cb.assign(i, 0), i < len, cb.assign(i, i + 1), skip(cb, r, in))
    } else {
      val nMissing = cb.newLocal[Int]("nMissing", UnsafeUtils.packBitsToBytes(len))
      val mbytes = cb.newLocal[Long]("mbytes", r.allocate(const(1L), nMissing.toL))
      cb += in.readBytes(r, mbytes, nMissing)
      cb.forLoop(cb.assign(i, 0), i < len, cb.assign(i, i + 1),
        cb.ifx(!Region.loadBit(mbytes, i.toL), skip(cb, r, in)))
    }
  }

  def _asIdent = s"array_of_${elementType.asIdent}"
  def _toPretty = s"EArray[$elementType]"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false) {
    sb.append("EArray[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }

  def setRequired(newRequired: Boolean): EArray = EArray(elementType, newRequired)
}
