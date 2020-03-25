package is.hail.expr.types.encoded

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.BaseType
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

final case class EArray(val elementType: EType, override val required: Boolean = false) extends EContainer {
  override def _decodeCompatible(pt: PType): Boolean = {
    pt.required == required &&
      pt.isInstanceOf[PArray] &&
      elementType.decodeCompatible(pt.asInstanceOf[PArray].elementType)
  }

  override def _encodeCompatible(pt: PType): Boolean = {
    pt.required == required &&
      pt.isInstanceOf[PArray] &&
      elementType.encodeCompatible(pt.asInstanceOf[PArray].elementType)
  }

  def _decodedPType(requestedType: Type): PType = {
    val elementPType = elementType.decodedPType(requestedType.asInstanceOf[TContainer].elementType)
    requestedType match {
      case t: TSet =>
        PSet(elementPType, required)
      case t: TArray =>
        PArray(elementPType, required)
      case t: TDict =>
        val et = elementPType.asInstanceOf[PStruct]
        PDict(et.fieldType("key"), et.fieldType("value"), required)
    }
  }

  def buildPrefixEncoder(pt: PArray, mb: EmitMethodBuilder[_], array: Value[Long],
    out: Value[OutputBuffer], prefixLength: Code[Int]
  ): Code[Unit] = {
    val len = mb.newLocal[Int]("len")
    val prefixLen = mb.newLocal[Int]("prefixLen")

    val writeLen = out.writeInt(prefixLen)
    val writeMissingBytes =
      if (!pt.elementType.required) {
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

    val i = mb.newLocal[Int]("i")

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

  def _buildEncoder(pt: PType, mb: EmitMethodBuilder[_], v: Value[_], out: Value[OutputBuffer]): Code[Unit] = {
    val pa = pt.asInstanceOf[PArray]
    val array = coerce[Long](v)
    buildPrefixEncoder(pa, mb, array, out, pa.loadLength(array))
  }

  def _buildDecoder(
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

    Code(
      len := in.readInt(),
      array := t.allocate(region, len),
      t.storeLength(array, len),
      if (elementType.required) {
        assert(t.elementType.required) // XXX For now
        Code._empty
      } else
        in.readBytes(region, array + const(t.lengthHeaderBytes), t.nMissingBytes(len)),
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
}
