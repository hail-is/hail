package is.hail.types.encoded

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.BaseType
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

final case class EPackedIntArray(
  override val required: Boolean = false,
  elementsRequired: Boolean
) extends EContainer with EFundamentalType {
  def elementType: EType = EInt32(elementsRequired)

  override def _compatible(pt: PType): Boolean = {
    pt.required == required &&
      pt.isInstanceOf[PArray] &&
      EInt32(elementsRequired).decodeCompatible(pt.asInstanceOf[PArray].elementType)
  }

  def _decodedPType(requestedType: Type): PType = EArray(EInt32(elementsRequired), required)._decodedPType(requestedType)


  def _buildFundamentalDecoder(pt: PType, mb: EmitMethodBuilder[_], region: Value[Region], in: Value[InputBuffer]): Code[_] = {
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

  def _buildFundamentalEncoder(cb: EmitCodeBuilder, pt: PType, v: Value[_], out: Value[OutputBuffer]): Unit = {
    val pa = pt.asInstanceOf[PArray]
    val array = coerce[Long](v)
    val len = cb.newLocal[Int]("len", pa.loadLength(array))
    val packer = cb.newLocal[IntPacker]("packer", getPacker(cb.emb))
    val keysLen = cb.newLocal[Int]("keysLen", (len + const(3)) / const(4))
    val dataLen = cb.newLocal[Int]("dataLen", len * const(4))
    val i = cb.newLocal[Int]("i", 0)

    cb += out.writeInt(len)
    if (!elementsRequired)
      cb += out.writeBytes(array + const(pa.lengthHeaderBytes), pa.nMissingBytes(len))
    cb += packer.load().ensureSpace(keysLen, dataLen)
    cb += packer.load().resetPack()
    cb.forLoop(cb.assign(i, 0), i < len, cb.assign(i, i + 1), {
      cb.ifx(pa.isElementDefined(array, i), {
        cb += packer.invoke[Long, Unit]("pack", pa.elementOffset(array, len, i))
      })
    })
    cb += packer.load().finish()
    cb += out.writeInt(packer.load().ki + packer.load().di)
    cb += out.write(packer.load().keys, const(0), packer.load().ki)
    cb += out.write(packer.load().data, const(0), packer.load().di)
  }

  def _asIdent: String = s"packedintarray_w_${if (elementsRequired) "required" else "optional"}_elements"
  def _toPretty: String = s"EPackedIntArray[${if (elementsRequired) "True" else "False"}]"

  private def getPacker(mb: EmitMethodBuilder[_]): Value[IntPacker] = {
    mb.getOrDefineLazyField[IntPacker](Code.newInstance[IntPacker], "thePacker")
  }

  def setRequired(newRequired: Boolean): EPackedIntArray = EPackedIntArray(newRequired, elementsRequired)
}
