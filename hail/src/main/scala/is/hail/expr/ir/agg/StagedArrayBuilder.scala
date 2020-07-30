package is.hail.expr.ir.agg

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitFunctionBuilder}
import is.hail.types.physical._
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.utils._

object StagedArrayBuilder {
  val END_SERIALIZATION: Int = 0x12345678
}

class StagedArrayBuilder(eltType: PType, cb: EmitClassBuilder[_], region: Value[Region], var initialCapacity: Int = 8) {
  val eltArray = PCanonicalArray(eltType.setRequired(false), required = true) // element type must be optional for serialization to work
  val stateType = PCanonicalTuple(true, PInt32Required, PInt32Required, eltArray)

  val size: Settable[Int] = cb.genFieldThisRef[Int]("size")
  private val capacity = cb.genFieldThisRef[Int]("capacity")
  val data = cb.genFieldThisRef[Long]("data")

  private val tmpOff = cb.genFieldThisRef[Long]("tmp_offset")
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
      data := StagedRegionValueBuilder.deepCopyFromOffset(cb, region, eltArray, Region.loadAddress(dataOffset(tmpOff))))
  }

  def reallocateData(): Code[Unit] = {
    data := StagedRegionValueBuilder.deepCopyFromOffset(cb, region, eltArray, data)
  }

  def storeTo(dest: Code[Long]): Code[Unit] = {
    Code(
      tmpOff := dest,
      Region.storeInt(currentSizeOffset(tmpOff), size),
      Region.storeInt(capacityOffset(tmpOff), capacity),
      Region.storeAddress(dataOffset(tmpOff), data)
    )
  }

  def serialize(codec: BufferSpec): Value[OutputBuffer] => Code[Unit] = {
    { ob: Value[OutputBuffer] =>
      val enc = TypedCodecSpec(eltArray, codec).buildTypedEmitEncoderF[Long](eltArray, cb)

      Code(
        ob.writeInt(size),
        ob.writeInt(capacity),
        enc(region, data, ob),
        ob.writeInt(const(StagedArrayBuilder.END_SERIALIZATION))
      )
    }
  }

  def deserialize(codec: BufferSpec): Value[InputBuffer] => Code[Unit] = {
    val (decType, dec) = TypedCodecSpec(eltArray, codec).buildEmitDecoderF[Long](cb)
    assert(decType == eltArray)

    { (ib: Value[InputBuffer]) =>
      Code(
        size := ib.readInt(),
        capacity := ib.readInt(),
        data := dec(region, ib),
        ib.readInt()
          .cne(const(StagedArrayBuilder.END_SERIALIZATION))
          .orEmpty(Code._fatal[Unit](s"StagedArrayBuilder serialization failed"))
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
        StagedRegionValueBuilder.deepCopy(cb, region, eltType, elt, dest)
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

  def elementOffset(idx: Value[Int]): (Code[Boolean], Code[Long]) =
    (eltArray.isElementMissing(data, idx), eltArray.elementOffset(data, capacity, idx))


  def loadElement(idx: Value[Int]): (Code[Boolean], Code[_]) = {
    val (m, off) = elementOffset(idx)
    (m, Region.loadIRIntermediate(eltType)(off))
  }

  private def resize(): Code[Unit] = {
    val newDataOffset = cb.genFieldThisRef[Long]("new_data_offset")
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
