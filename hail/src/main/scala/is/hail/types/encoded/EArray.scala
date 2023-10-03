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
    val elemAddr = cb.newLocal[Long]("elemAddr", arrayType.firstElementOffset(array, len))
    arrayType.storeLength(cb, array, len)

    val i = cb.newLocal[Int]("i")
    val mbyteOffset = cb.newLocal[Long]("mbyteOffset", array + arrayType.lengthHeaderBytes)
    val mbyte = cb.newLocal[Byte]("mbyte", 0.toByte)
    val presentBitsLong = cb.newLocal[Long]("presentBitsLong", 0)
    val inBlockIndexToPresentValue = cb.newLocal[Int]("inBlockIndexToPresentValue", 0)
    val readElemF = elementType.buildInplaceDecoder(arrayType.elementType, cb.emb.ecb)

    if (!elementType.required)
      cb += in.readBytes(region, array + const(arrayType.lengthHeaderBytes), arrayType.nMissingBytes(len))

    val LONG_MASK_ALL_BUT_FIRST = const(0x7fffffffffffffffL)

    def numberOfLeadingZeros(l: Code[Long]): Code[Int] =
      Code.invokeStatic1[java.lang.Long, Long, Int]("numberOfLeadingZeros", l)

    cb.assign(i, 0)
    if (elementType.required) {
      cb.forLoop({}, i + 64 < len, cb.assign(i, i + 64), {
        for (_ <- 0 to 63) {
          readElemF(cb, region, elemAddr, in)
          cb.assign(elemAddr, arrayType.nextElementAddress(elemAddr))
        }
      })
    } else {
      cb.forLoop({}, i + 64 < len, cb.assign(i, i + 64), {
        cb.assign(presentBitsLong, ~Region.loadLong(mbyteOffset))
        cb.assign(mbyteOffset, mbyteOffset + 8)
        cb.assign(inBlockIndexToPresentValue, numberOfLeadingZeros(presentBitsLong))

        cb.whileLoop(inBlockIndexToPresentValue < 64, {
          cb.assign(elemAddr,
            arrayType.elementOffset(array, len, i + inBlockIndexToPresentValue))
          readElemF(cb, region, elemAddr, in)

          cb.assign(inBlockIndexToPresentValue,
            numberOfLeadingZeros(presentBitsLong & (LONG_MASK_ALL_BUT_FIRST >>> inBlockIndexToPresentValue)))
        })
      })
      cb.assign(elemAddr, arrayType.elementOffset(array, len, i))
    }
    cb.forLoop({}, i < len,
      {
        cb.assign(i, i + 1)
        cb.assign(elemAddr, arrayType.nextElementAddress(elemAddr))
      },
      {
        if (elementType.required) {
          readElemF(cb, region, elemAddr, in)
        } else {
          cb.ifx((i % 8).ceq(0),
            {
              cb.assign(mbyte, Region.loadByte(mbyteOffset))
              cb.assign(mbyteOffset, mbyteOffset + 1)
            }
          )
          cb.ifx((mbyte.load() & (const(1) << (i & 7))).ceq(0),
            readElemF(cb, region, elemAddr, in))
        }
      }
    )

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
