package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder}
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.types.physical._
import is.hail.types.physical.stypes.SCode
import is.hail.utils._

object StagedArrayBuilder {
  val END_SERIALIZATION: Int = 0x12345678
}

class StagedArrayBuilder(eltType: PType, kb: EmitClassBuilder[_], region: Value[Region], var initialCapacity: Int = 8) {
  val eltArray = PCanonicalArray(eltType.setRequired(false), required = true) // element type must be optional for serialization to work
  val stateType = PCanonicalTuple(true, PInt32Required, PInt32Required, eltArray)

  val size: Settable[Int] = kb.genFieldThisRef[Int]("size")
  private val capacity = kb.genFieldThisRef[Int]("capacity")
  val data = kb.genFieldThisRef[Long]("data")

  private val tmpOff = kb.genFieldThisRef[Long]("tmp_offset")
  private val currentSizeOffset: Code[Long] => Code[Long] = stateType.fieldOffset(_, 0)
  private val capacityOffset: Code[Long] => Code[Long] = stateType.fieldOffset(_, 1)
  private val dataOffset: Code[Long] => Code[Long] = stateType.fieldOffset(_, 2)

  def loadFrom(cb: EmitCodeBuilder, src: Code[Long]): Unit = {
    cb.assign(tmpOff, src)
    cb.assign(size, Region.loadInt(currentSizeOffset(tmpOff)))
    cb.assign(capacity, Region.loadInt(capacityOffset(tmpOff)))
    cb.assign(data, Region.loadAddress(dataOffset(tmpOff)))
  }


  def copyFrom(cb: EmitCodeBuilder, src: Code[Long]): Unit = {
    cb.assign(tmpOff, src)
    cb.assign(size, Region.loadInt(currentSizeOffset(tmpOff)))
    cb.assign(capacity, Region.loadInt(capacityOffset(tmpOff)))
    cb.assign(data, eltArray.store(cb, region, eltArray.loadCheapPCode(cb, Region.loadAddress(dataOffset(tmpOff))), deepCopy = true))
  }

  def reallocateData(cb: EmitCodeBuilder): Unit = {
    cb.assign(data, eltArray.store(cb, region, eltArray.loadCheapPCode(cb, data), deepCopy = true))
  }

  def storeTo(cb: EmitCodeBuilder, dest: Code[Long]): Unit = {
    cb.assign(tmpOff, dest)
    cb += Region.storeInt(currentSizeOffset(tmpOff), size)
    cb += Region.storeInt(capacityOffset(tmpOff), capacity)
    cb += Region.storeAddress(dataOffset(tmpOff), data)
  }

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    val enc = TypedCodecSpec(eltArray, codec).buildTypedEmitEncoderF[Long](eltArray, kb)
    (cb: EmitCodeBuilder, ob: Value[OutputBuffer]) => {

      cb += ob.writeInt(size)
      cb += ob.writeInt(capacity)
      cb += enc(region, data, ob)
      cb += ob.writeInt(const(StagedArrayBuilder.END_SERIALIZATION))
    }
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    val (decType, dec) = TypedCodecSpec(eltArray, codec).buildEmitDecoderF[Long](kb)
    assert(decType == eltArray)

    (cb: EmitCodeBuilder, ib: Value[InputBuffer]) => {
      cb.assign(size, ib.readInt())
      cb.assign(capacity, ib.readInt())
      cb.assign(data, dec(region, ib))
      cb += ib.readInt()
        .cne(const(StagedArrayBuilder.END_SERIALIZATION))
        .orEmpty(Code._fatal[Unit](s"StagedArrayBuilder serialization failed"))
    }
  }

  private def incrementSize(cb: EmitCodeBuilder): Unit = {
    cb.assign(size, size + 1)
    resize(cb)
  }

  def setMissing(cb: EmitCodeBuilder): Unit = incrementSize(cb) // all elements set to missing on initialization


  def append(cb: EmitCodeBuilder, elt: SCode, deepCopy: Boolean = true): Unit = {
    cb += eltArray.setElementPresent(data, size)
    eltType.storeAtAddress(cb, eltArray.elementOffset(data, capacity, size), region, elt, deepCopy)
    incrementSize(cb)
  }

  def initializeWithCapacity(cb: EmitCodeBuilder, capacity: Code[Int]): Unit = initialize(cb, 0, capacity)

  def initialize(cb: EmitCodeBuilder): Unit = initialize(cb, const(0), const(initialCapacity))

  private def initialize(cb: EmitCodeBuilder, _size: Code[Int], _capacity: Code[Int]): Unit = {
    cb += Code(
      size := _size,
      capacity := _capacity,
      data := eltArray.allocate(region, capacity),
      eltArray.stagedInitialize(data, capacity, setMissing = true)
    )
  }

  def elementOffset(idx: Value[Int]): Code[Long] = eltArray.elementOffset(data, capacity, idx)


  def loadElement(cb: EmitCodeBuilder, idx: Value[Int]): EmitCode = {
    val m = eltArray.isElementMissing(data, idx)
    EmitCode(Code._empty, m, eltType.loadCheapPCode(cb, eltArray.loadElement(data, capacity, idx)))
  }

  private def resize(cb: EmitCodeBuilder): Unit = {
    val newDataOffset = kb.genFieldThisRef[Long]("new_data_offset")
    cb.ifx(size.ceq(capacity),
      {
        cb.assign(capacity, capacity * 2)
        cb.assign(newDataOffset, eltArray.allocate(region, capacity))
        cb += eltArray.stagedInitialize(newDataOffset, capacity, setMissing = true)
        cb += Region.copyFrom(data + eltArray.lengthHeaderBytes, newDataOffset + eltArray.lengthHeaderBytes, eltArray.nMissingBytes(size).toL)
        cb += Region.copyFrom(data + eltArray.elementsOffset(size),
          newDataOffset + eltArray.elementsOffset(capacity.load()),
          size.toL * const(eltArray.elementByteSize))
        cb.assign(data, newDataOffset)
      })
  }
}
