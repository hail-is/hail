package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitMethodBuilder, EmitRegion, EmitTriplet}
import is.hail.expr.types.physical.{PArray, PBinary, PBoolean, PFloat32, PFloat64, PInt32, PInt32Required, PInt64, PInt64Required, PString, PTuple, PType}
import is.hail.io.{CodecSpec, InputBuffer, OutputBuffer}
import is.hail.utils._

object StagedArrayBuilder {
  val BASE_CAPACITY: Int = 8
  val END_SERIALIZATION: Int = 0x12345678
}

class StagedArrayBuilder(eltType: PType, mb: EmitMethodBuilder, er: EmitRegion, region: Code[Region]) {
  val stateType = PTuple(true, PInt32Required, PInt32Required, PInt64Required, PInt64Required)
  val size: ClassFieldRef[Int] = mb.newField[Int]
  private val capacity = mb.newField[Int]
  private val data = mb.newField[Long]
  private val missingBits = mb.newField[Long]

  private val currentSizeOffset: Code[Long] => Code[Long] = stateType.loadField(_, 0)
  private val capacityOffset: Code[Long] => Code[Long] = stateType.loadField(_, 1)
  private val dataOffset: Code[Long] => Code[Long] = stateType.loadField(_, 2)
  private val missingOffset: Code[Long] => Code[Long] = stateType.loadField(_, 3)

  def alignedByteSize(alignment: Long, byteSize: Long): Long = {
    if (byteSize % alignment == 0)
      byteSize
    else
      byteSize + (alignment - byteSize % alignment)
  }

  val alignedElementByteSize: Long = alignedByteSize(eltType.alignment, eltType.byteSize)
  
  def loadFields(src: Code[Long]): Code[Unit] = {
    Code(
      size := region.loadInt(currentSizeOffset(src)),
      capacity := region.loadInt(capacityOffset(src)),
      data := region.loadAddress(dataOffset(src)),
      missingBits := region.loadAddress(missingOffset(src))
    )
  }

  def storeFields(dest: Code[Long]): Code[Unit] = {
    Code(
      region.storeInt(currentSizeOffset(dest), size),
      region.storeInt(capacityOffset(dest), capacity),
      region.storeAddress(dataOffset(dest), data),
      region.storeAddress(missingOffset(dest), missingBits)
    )
  }

  def serialize(codec: CodecSpec): Code[OutputBuffer] => Code[Unit] = {
    { ob: Code[OutputBuffer] =>
      val eltEnc: (Code[Long]) => Code[Unit] =
        eltType.fundamentalType match {
          case _: PBoolean => (offset: Code[Long]) => ob.writeBoolean(region.loadBoolean(offset))
          case _: PInt32 => (offset: Code[Long]) => ob.writeInt(region.loadInt(offset))
          case _: PInt64 => (offset: Code[Long]) => ob.writeLong(region.loadLong(offset))
          case _: PFloat32 => (offset: Code[Long]) => ob.writeFloat(region.loadFloat(offset))
          case _: PFloat64 => (offset: Code[Long]) => ob.writeDouble(region.loadDouble(offset))
          case _ =>
            val enc = codec.buildEmitEncoderF[Long](eltType, eltType, mb.fb)
            (offset: Code[Long]) => enc(region, offset, ob)
        }

      val i = mb.newLocal[Int]
      val (eltMissing, elt) = loadElementOffset(i)

      Code(
        ob.writeInt(size),
        ob.writeInt(capacity),
        i := 0,
        Code.whileLoop(i < size,
          eltMissing.mux(
            ob.writeByte(const(1.toByte)),
            Code(
              ob.writeByte(const(0.toByte)),
              eltEnc(elt))),
          i := i + 1
        ),
        ob.writeInt(const(StagedArrayBuilder.END_SERIALIZATION))
      )
    }
  }

  def deserialize(codec: CodecSpec): Code[InputBuffer] => Code[Unit] = {
    val eltDec =
      eltType.fundamentalType match {
        case _: PBoolean => (offset: Code[Long], ib: Code[InputBuffer]) => region.storeBoolean(offset, ib.readBoolean())
        case _: PInt32 => (offset: Code[Long], ib: Code[InputBuffer]) => region.storeInt(offset, ib.readInt())
        case _: PInt64 => (offset: Code[Long], ib: Code[InputBuffer]) => region.storeLong(offset, ib.readLong())
        case _: PFloat32 => (offset: Code[Long], ib: Code[InputBuffer]) => region.storeFloat(offset, ib.readFloat())
        case _: PFloat64 => (offset: Code[Long], ib: Code[InputBuffer]) => region.storeDouble(offset, ib.readDouble())
        case _: PBinary | _: PArray =>
          val eltDec = codec.buildEmitDecoderF[Long](eltType, eltType, mb.fb)
          (offset: Code[Long], ib: Code[InputBuffer]) =>
            region.storeAddress(offset, eltDec(region, ib))
        case _ =>
          val eltDec = codec.buildEmitDecoderF[Long](eltType, eltType, mb.fb)
          (offset: Code[Long], ib: Code[InputBuffer]) =>
            region.copyFrom(region, eltDec(region, ib), offset, eltType.byteSize)
      }

    val i = mb.newLocal[Int]

    { ib: Code[InputBuffer] =>
      Code(
        initialize(ib.readInt(), ib.readInt()),
        i := 0,
        Code.whileLoop(i < size,
          ib.readByte().cne(const(0)).mux(
            region.storeByte(missingBits + i.toL, const(1)),
            Code(
              region.storeByte(missingBits + i.toL, const(0)),
              eltDec(data + (i.toL * const(alignedElementByteSize)), ib)
            )
          ),
          i += 1),
        ib.readInt()
          .cne(const(StagedArrayBuilder.END_SERIALIZATION))
          .orEmpty(Code._fatal(s"StagedArrayBuilder serialization failed"))
      )
    }
  }

  private def incrementSize(): Code[Unit] = Code(
    size := size + 1,
    resize()
  )

  def setMissing(): Code[Unit] = {
    Code(
      //      dump("setMissing"),
      region.storeByte(missingBits + size.toL, const(1)),
      incrementSize()
    )
  }

  def append(elt: Code[_]): Code[Unit] = {
    val dest = data + (size.toL * const(alignedElementByteSize))
    Code(
      region.storeByte(missingBits + size.toL, const(0)),
      eltType.fundamentalType match {
        case _: PBoolean => region.storeByte(dest, coerce[Boolean](elt).toI.toB)
        case _: PInt32 => region.storeInt(dest, coerce[Int](elt))
        case _: PInt64 => region.storeLong(dest, coerce[Long](elt))
        case _: PFloat32 => region.storeFloat(dest, coerce[Float](elt))
        case _: PFloat64 => region.storeDouble(dest, coerce[Double](elt))
        case _ => StagedRegionValueBuilder.deepCopy(er, eltType, coerce[Long](elt), dest)
      }, incrementSize())
  }

  def initialize(): Code[Unit] = initialize(const(0), const(StagedArrayBuilder.BASE_CAPACITY))

  private def initialize(_size: Code[Int], _capacity: Code[Int]): Code[Unit] = {
    Code(
      size := _size,
      capacity := _capacity,
      data := region.allocate(const(eltType.alignment), capacity.toL * const(alignedElementByteSize)),
      missingBits := region.allocate(const(1L), capacity.toL)
    )
  }

  def loadElementOffset(idx: Code[Int]): (Code[Boolean], Code[Long]) = {
    val f: Code[Long] => Code[Long] = if (!eltType.isPrimitive)
      (o: Code[Long]) => coerce[Long](region.loadIRIntermediate(eltType)(o))
    else identity[Code[Long]]
    (region.loadByte(missingBits + idx.toL).ceq(const(1.toByte)),
      coerce[Long](f(data + (idx.toL * alignedElementByteSize))))
  }

  def loadElement[T](idx: Code[Int]): (Code[Boolean], Code[T]) = {
    (region.loadByte(missingBits + idx.toL).ceq(const(1.toByte)),
      coerce[T](region.loadIRIntermediate(eltType)(data + (idx.toL * alignedElementByteSize))))
  }

  private def resize(): Code[Unit] = {
    val newDataOffset = mb.newLocal[Long]
    val newMissingOffset = mb.newLocal[Long]
    size.ceq(capacity)
      .orEmpty(
        Code(
          newDataOffset := region.allocate(const(eltType.alignment), const(alignedElementByteSize * 2L) * size.toL),
          region.copyFrom(region, data, newDataOffset, size.toL * const(alignedElementByteSize)),
          data := newDataOffset,
          newMissingOffset := region.allocate(const(1L), const(2L) * size.toL),
          region.copyFrom(region, missingBits, newMissingOffset, size.toL),
          missingBits := newMissingOffset,
          capacity := capacity * 2
        )
      )
  }
}
