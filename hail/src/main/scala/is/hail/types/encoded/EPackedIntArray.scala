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


  def _buildFundamentalDecoder(
    cb: EmitCodeBuilder,
    pt: PType,
    region: Value[Region],
    in: Value[InputBuffer]
  )(implicit line: LineNumber): Code[_] = {
    val pa = pt.asInstanceOf[PArray]

    val len = cb.newLocal[Int]("len", in.readInt())
    val array = cb.newLocal[Long]("array", pa.allocate(region, len))
    cb += pa.storeLength(array, len)
    if (!elementsRequired)
      cb += in.readBytes(region, array + const(pa.lengthHeaderBytes), pa.nMissingBytes(len))
    val dlen = cb.newLocal[Int]("dlen", in.readInt())

    val i = cb.newLocal[Int]("i")
    val n = cb.newLocal[Int]("n", 0)
    if (elementsRequired)
      cb.assign(n, len)
    else
      cb.forLoop(cb.assign(i, 0), i < len, cb.assign(i, i + 1),
        cb.assign(n, n + pa.isElementDefined(array, i).toI))

    val klen = cb.newLocal[Int]("klen", (n + const(3)) / const(4))
    cb.assign(dlen, dlen - klen)
    val unpacker = cb.newLocal[IntPacker]("unpacker", getPacker(cb.emb))
    cb += unpacker.load().ensureSpace(klen, dlen)
    cb += in.read(unpacker.load().keys, 0, klen)
    cb += in.read(unpacker.load().data, 0, dlen)
    cb += unpacker.load().resetUnpack()
    cb.forLoop(cb.assign(i, 0), i < len, cb.assign(i, i + 1), {
      val unpack = unpacker.invoke[Long, Unit]("unpack", pa.elementOffset(array, len, i))
      if (elementsRequired)
        cb += unpack
      else
        cb.ifx(pa.isElementDefined(array, i), cb += unpack)
    })

    array
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer])(implicit line: LineNumber): Unit = {
    val len = cb.newLocal[Int]("len", in.readInt())
    if (!elementsRequired) cb += in.skipBytes(UnsafeUtils.packBitsToBytes(len))
    cb.assign(len, in.readInt())
    cb += in.skipBytes(len)
  }

  def _buildFundamentalEncoder(cb: EmitCodeBuilder, pt: PType, v: Value[_], out: Value[OutputBuffer])(implicit line: LineNumber): Unit = {
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

  private def getPacker(mb: EmitMethodBuilder[_])(implicit line: LineNumber): Value[IntPacker] = {
    mb.getOrDefineLazyField[IntPacker](Code.newInstance[IntPacker], "thePacker")
  }

  def setRequired(newRequired: Boolean): EPackedIntArray = EPackedIntArray(newRequired, elementsRequired)
}
