package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.EmitFunctionBuilder
import is.hail.expr.types.physical._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.utils._

object StagedArrayBuilder {
  val END_SERIALIZATION: Int = 0x12345678
}

class StagedArrayBuilder(eltType: PType, fb: EmitFunctionBuilder[_], region: Code[Region], var initialCapacity: Int = 8) {
  val eltArray = PArray(eltType.setRequired(false), required = true) // element type must be optional for serialization to work
  val stateType = PTuple(true, PInt32Required, PInt32Required, eltArray)

  val size: ClassFieldRef[Int] = fb.newField[Int]("size")
  private val capacity = fb.newField[Int]("capacity")
  val data = fb.newField[Long]("data")

  private val tmpOff = fb.newField[Long]("tmp_offset")
  private val currentSizeOffset: Code[Long] => Code[Long] = stateType.fieldOffset(_, 0)
  private val capacityOffset: Code[Long] => Code[Long] = stateType.fieldOffset(_, 1)
  private val dataOffset: Code[Long] => Code[Long] = stateType.fieldOffset(_, 2)

  def loadFrom(src: Code[Long]): Code[Unit] = {
    Code(
      tmpOff := src,
      size := Region.loadInt(currentSizeOffset(tmpOff)),
      capacity := Region.loadInt(capacityOffset(tmpOff)),
      data := Region.loadAddress(dataOffset(tmpOff))
    )
  }


  def copyFrom(src: Code[Long]): Code[Unit] = {
    Code(
      tmpOff := src,
      size := Region.loadInt(currentSizeOffset(tmpOff)),
      capacity := Region.loadInt(capacityOffset(tmpOff)),
      data := StagedRegionValueBuilder.deepCopyFromOffset(fb, region, eltArray, Region.loadAddress(dataOffset(tmpOff))))
  }

  def reallocateData(): Code[Unit] = {
    data := StagedRegionValueBuilder.deepCopyFromOffset(fb, region, eltArray, data)
  }

  def storeTo(dest: Code[Long]): Code[Unit] = {
    Code(
      tmpOff := dest,
      Region.storeInt(currentSizeOffset(tmpOff), size),
      Region.storeInt(capacityOffset(tmpOff), capacity),
      Region.storeAddress(dataOffset(tmpOff), data)
    )
  }

  def serialize(codec: BufferSpec): Code[OutputBuffer] => Code[Unit] = {
    { ob: Code[OutputBuffer] =>
      val enc = TypedCodecSpec(eltArray, codec).buildEmitEncoderF[Long](eltArray, fb)

      Code(
        ob.writeInt(size),
        ob.writeInt(capacity),
        enc(region, data, ob),
        ob.writeInt(const(StagedArrayBuilder.END_SERIALIZATION))
      )
    }
  }

  def deserialize(codec: BufferSpec): Code[InputBuffer] => Code[Unit] = {
    val (decType, dec) = TypedCodecSpec(eltArray, codec).buildEmitDecoderF[Long](eltArray.virtualType, fb)
    assert(decType == eltArray)

    { (ib: Code[InputBuffer]) =>
      Code(
        size := ib.readInt(),
        capacity := ib.readInt(),
        data := dec(region, ib),
        ib.readInt()
          .cne(const(StagedArrayBuilder.END_SERIALIZATION))
          .orEmpty[Unit](Code._fatal(s"StagedArrayBuilder serialization failed"))
      )
    }
  }

  private def incrementSize(): Code[Unit] = Code(
    size := size + 1,
    resize()
  )

  def setMissing(): Code[Unit] = incrementSize() // all elements set to missing on initialization


  def append(elt: Code[_], deepCopy: Boolean = true): Code[Unit] = {
    val dest = eltArray.elementOffset(data, capacity, size)
    Code(
      eltArray.setElementPresent(data, size),
      (if (deepCopy)
        StagedRegionValueBuilder.deepCopy(fb, region, eltType, elt, dest)
      else
        Region.storeIRIntermediate(eltType)(dest, elt)),
      incrementSize())
  }

  def initializeWithCapacity(capacity: Code[Int]): Code[Unit] = initialize(0, capacity)

  def initialize(): Code[Unit] = initialize(const(0), const(initialCapacity))

  private def initialize(_size: Code[Int], _capacity: Code[Int]): Code[Unit] = {
    Code(
      size := _size,
      capacity := _capacity,
      data := eltArray.allocate(region, capacity),
      eltArray.stagedInitialize(data, capacity, setMissing = true)
    )
  }

  def elementOffset(idx: Code[Int]): (Code[Boolean], Code[Long]) = {
    (eltArray.isElementMissing(data, idx), eltArray.elementOffset(data, capacity, idx))
  }

  def loadElement(idx: Code[Int]): (Code[Boolean], Code[_]) = {
    val (m, off) = elementOffset(idx)
    (m, Region.loadIRIntermediate(eltType)(off))
  }

  private def resize(): Code[Unit] = {
    val newDataOffset = fb.newField[Long]("new_data_offset")
    size.ceq(capacity)
      .orEmpty(
        Code(
          capacity := capacity * 2,
          newDataOffset := eltArray.allocate(region, capacity),
          eltArray.stagedInitialize(newDataOffset, capacity, setMissing = true),
          Region.copyFrom(data + eltArray.lengthHeaderBytes, newDataOffset + eltArray.lengthHeaderBytes, eltArray.nMissingBytes(size).toL),
          Region.copyFrom(data + eltArray.elementsOffset(size),
            newDataOffset + eltArray.elementsOffset(capacity.load()),
            size.toL * const(eltArray.elementByteSize)),
          data := newDataOffset
        )
      )
  }
}
