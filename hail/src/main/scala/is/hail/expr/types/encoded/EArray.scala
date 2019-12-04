package is.hail.expr.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.BaseType
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

final case class EArray(elementType: EType, override val required: Boolean = false) extends EType {
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

  def buildPrefixEncoder(pt: PArray, mb: EmitMethodBuilder, array: Code[Long],
    out: Code[OutputBuffer], prefixLength: Code[Int]
  ): Code[Unit] = {
    val len = mb.newLocal[Int]("len")
    val prefixLen = mb.newLocal[Int]("prefixLen")

    val writeLen = out.writeInt(prefixLen)
    val writeMissingBytes =
      if (!pt.elementType.required) {
        out.writeBytes(array + const(pt.lengthHeaderBytes), pt.nMissingBytes(prefixLen))
      } else
        Code._empty[Unit]

    val i = mb.newLocal[Int]("i")

    val writeElemF = elementType.buildEncoder(pt.elementType, mb)
    val elem = pt.elementOffset(array, len, i)
    val writeElems = Code(
      i := 0,
      Code.whileLoop(
        i < prefixLen,
        Code(
          pt.isElementDefined(array, i).mux(
            writeElemF(Region.loadIRIntermediate(pt.elementType)(elem), out), // XXX, get or loadIRIntermediate
            Code._empty[Unit]),
          i := i + const(1))))

    Code(
      len := pt.loadLength(array),
      prefixLen := prefixLength,
      writeLen,
      writeMissingBytes,
      writeElems)
  }

  def _buildEncoder(pt: PType, mb: EmitMethodBuilder, v: Code[_], out: Code[OutputBuffer]): Code[Unit] = {
    val pa = pt.asInstanceOf[PArray]
    val array = coerce[Long](v)
    buildPrefixEncoder(pa, mb, array, out, pa.loadLength(array))
  }

  def _buildDecoder(
    pt: PType,
    mb: EmitMethodBuilder,
    region: Code[Region],
    in: Code[InputBuffer]
  ): Code[Long] = {
    val t = pt.asInstanceOf[PArray]
    val len = mb.newLocal[Int]("len")
    val i = mb.newLocal[Int]("i")
    val array = mb.newLocal[Long]("array")
    val readElemF = elementType.buildInplaceDecoder(t.elementType, mb.fb)

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

  def _buildSkip(mb: EmitMethodBuilder, r: Code[Region], in: Code[InputBuffer]): Code[Unit] = {
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
        nMissing := PContainer.nMissingBytes(len),
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
