package is.hail.types.encoded

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.BaseType
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

final case class EArray(val elementType: EType, override val required: Boolean = false) extends EContainer with EFundamentalType {
  override def _decodeCompatible(pt: PType): Boolean = {
    pt.required <= required &&
      pt.isInstanceOf[PArray] &&
      elementType.decodeCompatible(pt.asInstanceOf[PArray].elementType)
  }

  override def _encodeCompatible(pt: PType): Boolean = {
    pt.required >= required &&
      pt.isInstanceOf[PArray] &&
      elementType.encodeCompatible(pt.asInstanceOf[PArray].elementType)
  }

  def _decodedPType(requestedType: Type): PType = {
    val elementPType = elementType.decodedPType(requestedType.asInstanceOf[TContainer].elementType)
    requestedType match {
      case _: TSet =>
        PCanonicalSet(elementPType, required)
      case _: TArray =>
        PCanonicalArray(elementPType, required)
      case _: TDict =>
        val et = elementPType.asInstanceOf[PStruct]
        PCanonicalDict(et.fieldType("key"), et.fieldType("value"), required)
    }
  }

  def buildPrefixEncoder(pt: PArray, mb: EmitMethodBuilder[_], array: Value[Long],
    out: Value[OutputBuffer], prefixLength: Code[Int]
  ): Code[Unit] = {
    val len = mb.newLocal[Int]("len")
    val prefixLen = mb.newLocal[Int]("prefixLen")

    val i = mb.newLocal[Int]("i")

    val writeLen = out.writeInt(prefixLen)
    val writeMissingBytes = pt match {
      case t: PCanonicalArray if t.elementType.required == elementType.required =>
        if (!elementType.required) {
          val nMissingLocal = mb.newLocal[Int]("nMissingBytes")
          Code(
            nMissingLocal := pt.nMissingBytes(prefixLen),
            (nMissingLocal > 0).orEmpty(
              Code(
                out.writeBytes(array + const(pt.lengthHeaderBytes), nMissingLocal - 1),
                out.writeByte((Region.loadByte(array + const(pt.lengthHeaderBytes) +
                  (nMissingLocal - 1).toL) & EType.lowBitMask(prefixLen)).toB)
              )
            )
          )
        } else
          Code._empty
      case _ =>
        val b = Code.newLocal[Int]("b")
        val shift = Code.newLocal[Int]("shift")
        EmitCodeBuilder.scopedVoid(mb) { cb =>
          cb.assign(i, 0)
          cb.assign(b, 0)
          cb.assign(shift, 0)
          cb.whileLoop(i < prefixLen, {
            cb.ifx(pt.isElementMissing(array, i), cb.assign(b, b | (const(1) << shift)))
            cb.assign(shift, shift + 1)
            cb.assign(i, i + 1)
            cb.ifx(shift.ceq(7), {
              cb.assign(shift, 0)
              cb += out.writeByte(b.toB)
              cb.assign(b, 0)
            })
          })
          cb.ifx(shift > 0, cb += out.writeByte(b.toB))
        }
    }


    val writeElemF = elementType.buildEncoder(pt.elementType, mb.ecb)
    val elem = pt.elementOffset(array, len, i)
    val writeElems = Code(
      i := 0,
      Code.whileLoop(
        i < prefixLen,
        Code(
          pt.isElementDefined(array, i).mux(
            writeElemF(Region.loadIRIntermediate(pt.elementType)(elem), out), // XXX, get or loadIRIntermediate
            Code._empty),
          i := i + const(1))))

    Code(
      len := pt.loadLength(array),
      prefixLen := prefixLength,
      writeLen,
      writeMissingBytes,
      writeElems)
  }

  def _buildFundamentalEncoder(pt: PType, mb: EmitMethodBuilder[_], v: Value[_], out: Value[OutputBuffer]): Code[Unit] = {
    val pa = pt.asInstanceOf[PArray]
    val array = coerce[Long](v)
    buildPrefixEncoder(pa, mb, array, out, pa.loadLength(array))
  }

  def _buildFundamentalDecoder(
    pt: PType,
    mb: EmitMethodBuilder[_],
    region: Value[Region],
    in: Value[InputBuffer]
  ): Code[Long] = {
    val t = pt.asInstanceOf[PArray]
    val len = mb.newLocal[Int]("len")
    val i = mb.newLocal[Int]("i")
    val array = mb.newLocal[Long]("array")
    val readElemF = elementType.buildInplaceDecoder(t.elementType, mb.ecb)

    val readMissing: Code[Unit] = pt match {
      case t: PCanonicalArray if t.elementType.required == elementType.required =>
        if (elementType.required)
          Code._empty
        else in.readBytes(region, array + const(t.lengthHeaderBytes), t.nMissingBytes(len))
      case _ =>
        if (elementType.required) {
          EmitCodeBuilder.scopedVoid(mb) { cb =>
            cb.assign(i, 0)
            cb.whileLoop(i < len, {
              cb += t.setElementPresent(array, i)
              cb.assign(i, i + 1)
            })
          }
        } else {
          val missingBitsAddr = mb.newLocal[Long]("missingBitsAddr")
          EmitCodeBuilder.scopedVoid(mb) { cb =>
            cb.assign(missingBitsAddr, region.allocate(1L, t.nMissingBytes(len).toL))
            cb.assign(i, 0)
            cb += in.readBytes(region, missingBitsAddr, t.nMissingBytes(len))
            cb.whileLoop(i < len, {
              cb.ifx(Region.loadBit(missingBitsAddr, i.toL),
                cb += t.setElementMissing(array, i),
                cb += t.setElementPresent(array, i))
              cb.assign(i, i + 1)
            })
          }
        }
    }

    Code(
      len := in.readInt(),
      array := t.allocate(region, len),
      t.storeLength(array, len),
      readMissing,
      i := 0,
      Code.whileLoop(
        i < len,
        Code(
          if (elementType.required)
            readElemF(region, t.elementOffset(array, len, i), in)
          else
            t.isElementDefined(array, i).mux(
              readElemF(region, t.elementOffset(array, len, i), in),
              Code._empty),
          i := i + const(1))),
      array.load())
  }

  def _buildSkip(mb: EmitMethodBuilder[_], r: Value[Region], in: Value[InputBuffer]): Code[Unit] = {
    val len = mb.newLocal[Int]("len")
    val i = mb.newLocal[Int]("i")
    val skip = elementType.buildSkip(mb)

    if (elementType.required) {
      Code(
        len := in.readInt(),
        i := 0,
        Code.whileLoop(i < len,
          Code(
            skip(r, in),
            i := i + const(1))))
    } else {
      val mbytes = mb.newLocal[Long]("mbytes")
      val nMissing = mb.newLocal[Int]("nMissing")
      Code(
        len := in.readInt(),
        nMissing := UnsafeUtils.packBitsToBytes(len),
        mbytes := r.allocate(const(1), nMissing.toL),
        in.readBytes(r, mbytes, nMissing),
        i := 0,
        Code.whileLoop(i < len,
          Region.loadBit(mbytes, i.toL).mux(
            Code._empty,
            skip(r, in)),
          i := i + const(1)))
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
