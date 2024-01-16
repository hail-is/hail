package is.hail.types.encoded

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical._
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SIndexablePointerValue}
import is.hail.types.physical.stypes.interfaces.SIndexableValue
import is.hail.types.virtual._
import is.hail.utils._

final case class EArray(val elementType: EType, override val required: Boolean = false)
    extends EContainer {
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

  def buildPrefixEncoder(
    cb: EmitCodeBuilder,
    value: SIndexableValue,
    out: Value[OutputBuffer],
    prefixLength: Code[Int],
  ): Unit = {
    val prefixLen = cb.newLocal[Int]("prefixLen", prefixLength)
    val i = cb.newLocal[Int]("i", 0)

    cb += out.writeInt(prefixLen)

    value.st match {
      case s @ SIndexablePointer(_: PCanonicalArray | _: PCanonicalSet | _: PCanonicalDict)
          if s.pType.elementType.required == elementType.required =>
        val pArray = s.pType match {
          case t: PCanonicalArray => t
          case t: PCanonicalSet => t.arrayRep
          case t: PCanonicalDict => t.arrayRep
        }

        val array = value.asInstanceOf[SIndexablePointerValue].a
        if (!elementType.required) {
          val nMissingBytes = cb.memoize(pArray.nMissingBytes(prefixLen), "nMissingBytes")
          cb.if_(
            nMissingBytes > 0, {
              cb += out.writeBytes(array + pArray.missingBytesOffset, nMissingBytes - 1)
              cb += out.writeByte((Region.loadByte(array + pArray.missingBytesOffset
                + (nMissingBytes - 1).toL) & EType.lowBitMask(prefixLen)).toB)
            },
          )
        }
      case _ =>
        if (elementType.required) {
          cb.if_(
            value.hasMissingValues(cb),
            cb._fatal("cannot encode indexable with missing element(s) to required EArray!"),
          )
        } else {
          val b = Code.newLocal[Int]("b")
          val shift = Code.newLocal[Int]("shift")
          cb.assign(b, 0)
          cb.assign(shift, 0)
          cb.while_(
            i < prefixLen, {
              cb.if_(value.isElementMissing(cb, i), cb.assign(b, b | (const(1) << shift)))
              cb.assign(shift, shift + 1)
              cb.assign(i, i + 1)
              cb.if_(
                shift.ceq(8), {
                  cb.assign(shift, 0)
                  cb += out.writeByte(b.toB)
                  cb.assign(b, 0)
                },
              )
            },
          )
          cb.if_(shift > 0, cb += out.writeByte(b.toB))
        }
    }

    cb.for_(
      cb.assign(i, 0),
      i < prefixLen,
      cb.assign(i, i + 1), {
        value.loadElement(cb, i).consume(
          cb,
          if (elementType.required)
            cb._fatal(s"required array element saw missing value at index ", i.toS, " in encode"),
          pc =>
            elementType.buildEncoder(pc.st, cb.emb.ecb)
              .apply(cb, pc, out),
        )
      },
    )
  }

  override def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit = {
    val ind = v.asInstanceOf[SIndexableValue]
    buildPrefixEncoder(cb, ind, out, ind.loadLength())
  }

  override def _buildDecoder(
    cb: EmitCodeBuilder,
    t: Type,
    region: Value[Region],
    in: Value[InputBuffer],
  ): SValue = {
    val st = decodedSType(t).asInstanceOf[SIndexablePointer]

    val arrayType: PCanonicalArray = st.pType match {
      case t: PCanonicalArray => t
      case t: PCanonicalSet => t.arrayRep
      case t: PCanonicalDict => t.arrayRep
    }

    assert(
      arrayType.elementType.required == elementType.required,
      s"${arrayType.elementType.required} | ${elementType.required}",
    )

    val len = cb.memoize(in.readInt(), "len")
    val array = cb.memoize(arrayType.allocate(region, len), "array")
    arrayType.storeLength(cb, array, len)

    val readElemF = elementType.buildInplaceDecoder(arrayType.elementType, cb.emb.ecb)

    val pastLastOff = cb.memoize(arrayType.pastLastElementOffset(array, len))
    if (elementType.required) {
      val elemOff = cb.newLocal[Long]("elemOff", arrayType.firstElementOffset(array, len))
      if (arrayType.zeroSizeElements) {
        // elements have 0 size, so all elements have the same address
        // still need to read `len` elements from the input stream, as they may have non-zero size
        val i = cb.newLocal[Int]("i")
        cb.for_(cb.assign(i, 0), i < len, cb.assign(i, i + 1), readElemF(cb, region, elemOff, in))
      } else {
        cb.for_(
          {},
          elemOff < pastLastOff,
          cb.assign(elemOff, arrayType.nextElementAddress(elemOff)),
          readElemF(cb, region, elemOff, in),
        )
      }
    } else {
      cb += in.readBytes(
        region,
        array + const(arrayType.missingBytesOffset),
        arrayType.nMissingBytes(len),
      )

      cb.if_(
        (len % 64).cne(0), {
          // ensure that the last missing block has all missing bits set past the last element
          val lastMissingBlockOff = cb.memoize(UnsafeUtils.roundDownAlignment(
            arrayType.pastLastMissingByteOff(array, len) - 1,
            8,
          ))
          val lastMissingBlock = cb.memoize(Region.loadLong(lastMissingBlockOff))
          cb += Region.storeLong(lastMissingBlockOff, lastMissingBlock | (const(-1L) << len))
        },
      )

      def unsetRightMostBit(x: Value[Long]): Code[Long] =
        x & (x - 1)

      val presentBits = cb.newLocal[Long]("presentBits", 0L)
      val mbyteOffset = cb.newLocal[Long]("mbyteOffset", array + arrayType.missingBytesOffset)
      val blockOff = cb.newLocal[Long]("blockOff", arrayType.firstElementOffset(array, len))
      val pastLastMissingByteOff =
        cb.memoize(arrayType.pastLastMissingByteOff(array, len), "pastLastMissingByteAddr")
      val inBlockIndexToPresentValue = cb.newLocal[Int]("inBlockIndexToPresentValue", 0)

      cb.for_(
        {},
        mbyteOffset < pastLastMissingByteOff, {
          cb.assign(blockOff, arrayType.incrementElementOffset(blockOff, 64))
          cb.assign(mbyteOffset, mbyteOffset + 8)
        }, {
          cb.assign(presentBits, ~Region.loadLong(mbyteOffset))
          cb.while_(
            presentBits.cne(0L), {
              cb.assign(inBlockIndexToPresentValue, presentBits.numberOfTrailingZeros)
              val elemOff = cb.memoize(
                arrayType.incrementElementOffset(blockOff, inBlockIndexToPresentValue)
              )
              readElemF(cb, region, elemOff, in)
              cb.assign(presentBits, unsetRightMostBit(presentBits))
            },
          )
        },
      )
    }

    new SIndexablePointerValue(st, array, len, cb.memoize(arrayType.firstElementOffset(array, len)))
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit = {
    val skip = elementType.buildSkip(cb.emb.ecb)
    val len = cb.newLocal[Int]("len", in.readInt())
    val i = cb.newLocal[Int]("i")
    if (elementType.required) {
      cb.for_(cb.assign(i, 0), i < len, cb.assign(i, i + 1), skip(cb, r, in))
    } else {
      val nMissing = cb.newLocal[Int]("nMissing", UnsafeUtils.packBitsToBytes(len))
      val mbytes = cb.newLocal[Long]("mbytes", r.allocate(const(1L), nMissing.toL))
      cb += in.readBytes(r, mbytes, nMissing)
      cb.for_(
        cb.assign(i, 0),
        i < len,
        cb.assign(i, i + 1),
        cb.if_(!Region.loadBit(mbytes, i.toL), skip(cb, r, in)),
      )
    }
  }

  def _asIdent = s"array_of_${elementType.asIdent}"
  def _toPretty = s"EArray[$elementType]"

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean = false): Unit = {
    sb.append("EArray[")
    elementType.pretty(sb, indent, compact)
    sb.append("]")
  }

  def setRequired(newRequired: Boolean): EArray = EArray(elementType, newRequired)
}
