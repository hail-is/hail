package is.hail.expr.types.encoded

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.BaseType
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

final case class EPackedIntArray(
  override val required: Boolean = false,
  val elementsRequired: Boolean
) extends EContainer {
  def elementType: EType = EInt32(elementsRequired)

  override def _compatible(pt: PType): Boolean = {
    pt.required == required &&
      pt.isInstanceOf[PArray] &&
      EInt32(elementsRequired).decodeCompatible(pt.asInstanceOf[PArray].elementType)
  }

  def _decodedPType(requestedType: Type): PType = EArray(EInt32(elementsRequired), required)._decodedPType(requestedType)

  def _buildDecoder(pt: PType, mb: EmitMethodBuilder[_], region: Value[Region], in: Value[InputBuffer]): Code[_] = {
    val pa = pt.asInstanceOf[PArray]

    val i = mb.newLocal[Int]("i")
    val len = mb.newLocal[Int]("len")
    val n = mb.newLocal[Int]("n")
    val dlen = mb.newLocal[Int]("dlen")
    val klen = mb.newLocal[Int]("klen")
    val array = mb.newLocal[Long]("array")
    val keys = mb.newLocal[Array[Byte]]("keys")
    val data = mb.newLocal[Array[Byte]]("data")
    val unpacker = mb.newLocal[IntPacker]("unpacker")

    Code(Code(FastIndexedSeq(
      unpacker := getPacker(mb),
      len := in.readInt(),
      array := pa.allocate(region, len),
      pa.storeLength(array, len),
      if (elementsRequired)
        Code._empty
      else
        in.readBytes(region, array + const(pa.lengthHeaderBytes), pa.nMissingBytes(len)),
      dlen := in.readInt(),
      if (elementsRequired)
        n := len
      else
        Code(
          i := 0,
          n := 0,
          Code.whileLoop(i < len,
            Code(
              n := n + pa.isElementDefined(array, i).toI,
              i := i + 1))),
      klen := (n + const(3)) / const(4),
      dlen := dlen - klen,
      unpacker.load().ensureSpace(klen, dlen),
      in.read(unpacker.load().keys, 0, klen),
      in.read(unpacker.load().data, 0, dlen),
      unpacker.load().resetUnpack(),
      i := 0,
      Code.whileLoop(i < len,
        Code(
          pa.isElementDefined(array, i).mux(
            unpacker.invoke[Long, Unit]("unpack", pa.elementOffset(array, len, i)),
            Code._empty),
          i := i + 1
        )))),
      array)
  }

  def _buildSkip(mb: EmitMethodBuilder[_], r: Value[Region], in: Value[InputBuffer]): Code[Unit] = {
    val len = mb.newLocal[Int]("len")

    Code(
      len := in.readInt(),
      if (elementsRequired)
        Code._empty
      else
        in.skipBytes(UnsafeUtils.packBitsToBytes(len)),
      len := in.readInt(),
      in.skipBytes(len)
    )
  }

  def _buildEncoder(pt: PType, mb: EmitMethodBuilder[_], v: Value[_], out: Value[OutputBuffer]): Code[Unit] = {
    val pa = pt.asInstanceOf[PArray]

    val packer = mb.newLocal[IntPacker]("packer")
    val len = mb.newLocal[Int]("len")
    val i = mb.newLocal[Int]("i")
    val array = coerce[Long](v)
    val keysLen = mb.newLocal[Int]("keysLen")
    val dataLen = mb.newLocal[Int]("dataLen")

    Code(Code(FastIndexedSeq(
      packer := getPacker(mb),
      len := pa.loadLength(array),
      out.writeInt(len),
      if (elementsRequired)
        Code._empty
      else
        out.writeBytes(array + const(pa.lengthHeaderBytes), pa.nMissingBytes(len)),
      keysLen := (len + const(3)) / const(4),
      dataLen := len * const(4),
      packer.load().ensureSpace(keysLen, dataLen),
      packer.load().resetPack(),
      i := 0,
      Code.whileLoop(i < len,
          Code(
            pa.isElementDefined(array, i).mux(
              packer.invoke[Long, Unit]("pack", pa.elementOffset(array, len, i)),
              Code._empty),
            i := i + const(1))),
      packer.load().finish(),
      out.writeInt(packer.load().ki + packer.load().di),
      out.write(packer.load().keys, const(0), packer.load().ki))),
      out.write(packer.load().data, const(0), packer.load().di))
  }

  def _asIdent: String = s"packedintarray_w_${if (elementsRequired) "required" else "optional"}_elements"
  def _toPretty: String = s"EPackedIntArray[${if (elementsRequired) "True" else "False"}]"

  private def getPacker(mb: EmitMethodBuilder[_]): Value[IntPacker] = {
    mb.getOrDefineLazyField[IntPacker](Code.newInstance[IntPacker], "thePacker")
  }
}
