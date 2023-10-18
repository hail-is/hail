package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder, EmitValue}
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.types.physical._
import is.hail.types.physical.stypes.SValue
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

  private val currentSizeOffset: Code[Long] => Code[Long] = stateType.fieldOffset(_, 0)
  private val capacityOffset: Code[Long] => Code[Long] = stateType.fieldOffset(_, 1)
  private val dataOffset: Code[Long] => Code[Long] = stateType.fieldOffset(_, 2)

  def loadFrom(cb: EmitCodeBuilder, src: Code[Long]): Unit = {
    val tmpOff = cb.memoize(src)
    cb.assign(size, Region.loadInt(currentSizeOffset(tmpOff)))
    cb.assign(capacity, Region.loadInt(capacityOffset(tmpOff)))
    cb.assign(data, Region.loadAddress(dataOffset(tmpOff)))
  }


  def cloneFrom(cb: EmitCodeBuilder, other: StagedArrayBuilder): Unit = {
    cb.assign(size, other.size)
    cb.assign(data, other.data)
    cb.assign(capacity, other.capacity)
  }

  def copyFrom(cb: EmitCodeBuilder, src: Code[Long]): Unit = {
    val tmpOff = cb.memoize(src)
    cb.assign(size, Region.loadInt(currentSizeOffset(tmpOff)))
    cb.assign(capacity, Region.loadInt(capacityOffset(tmpOff)))
    cb.assign(data, eltArray.store(cb, region, eltArray.loadCheapSCode(cb, Region.loadAddress(dataOffset(tmpOff))), deepCopy = true))
  }

  def reallocateData(cb: EmitCodeBuilder): Unit = {
    cb.assign(data, eltArray.store(cb, region, eltArray.loadCheapSCode(cb, data), deepCopy = true))
  }

  def storeTo(cb: EmitCodeBuilder, dest: Code[Long]): Unit = {
    val tmpOff = cb.memoize(dest)
    cb += Region.storeInt(currentSizeOffset(tmpOff), size)
    cb += Region.storeInt(capacityOffset(tmpOff), capacity)
    cb += Region.storeAddress(dataOffset(tmpOff), data)
  }

  def serialize(codec: BufferSpec): (EmitCodeBuilder, Value[OutputBuffer]) => Unit = {
    val codecSpec = TypedCodecSpec(eltArray, codec)
    (cb: EmitCodeBuilder, ob: Value[OutputBuffer]) => {

      cb += ob.writeInt(size)
      cb += ob.writeInt(capacity)
      codecSpec.encodedType.buildEncoder(eltArray.sType, kb)
        .apply(cb, eltArray.loadCheapSCode(cb, data), ob)
      cb += ob.writeInt(const(StagedArrayBuilder.END_SERIALIZATION))
    }
  }

  def deserialize(codec: BufferSpec): (EmitCodeBuilder, Value[InputBuffer]) => Unit = {
    val codecSpec = TypedCodecSpec(eltArray, codec)

    (cb: EmitCodeBuilder, ib: Value[InputBuffer]) => {
      cb.assign(size, ib.readInt())
      cb.assign(capacity, ib.readInt())

      val decValue = codecSpec.encodedType.buildDecoder(eltArray.virtualType, kb)
        .apply(cb, region, ib)
      cb.assign(data, eltArray.store(cb, region, decValue, deepCopy = false))

      cb.if_(ib.readInt() cne StagedArrayBuilder.END_SERIALIZATION,
        cb._fatal(s"StagedArrayBuilder serialization failed")
      )
    }
  }

  private def incrementSize(cb: EmitCodeBuilder): Unit = {
    cb.assign(size, size + 1)
    resize(cb)
  }

  def setMissing(cb: EmitCodeBuilder): Unit = incrementSize(cb) // all elements set to missing on initialization


  def append(cb: EmitCodeBuilder, elt: SValue, deepCopy: Boolean = true): Unit = {
    eltArray.setElementPresent(cb, data, size)
    eltType.storeAtAddress(cb, eltArray.elementOffset(data, capacity, size), region, elt, deepCopy)
    incrementSize(cb)
  }

  def overwrite(cb: EmitCodeBuilder, elt: EmitValue, idx: Value[Int], deepCopy: Boolean = true): Unit = {
    elt.toI(cb).consume(cb,
      PContainer.unsafeSetElementMissing(cb, eltArray, data, idx),
      value => eltType.storeAtAddress(cb, eltArray.elementOffset(data, capacity, idx), region, value, deepCopy))
  }

  def initializeWithCapacity(cb: EmitCodeBuilder, capacity: Code[Int]): Unit = initialize(cb, 0, capacity)

  def initialize(cb: EmitCodeBuilder): Unit = initialize(cb, const(0), const(initialCapacity))

  private def initialize(cb: EmitCodeBuilder, _size: Code[Int], _capacity: Code[Int]): Unit = {
    cb.assign(size, _size)
    cb.assign(capacity, _capacity)
    cb.assign(data, eltArray.allocate(region, capacity))
    eltArray.stagedInitialize(cb, data, capacity, setMissing = true)
  }

  def elementOffset(cb: EmitCodeBuilder, idx: Value[Int]): Value[Long] =
    cb.memoize(eltArray.elementOffset(data, capacity, idx))

  def loadElement(cb: EmitCodeBuilder, idx: Value[Int]): EmitCode = {
    val m = eltArray.isElementMissing(data, idx)
    EmitCode(Code._empty, m, eltType.loadCheapSCode(cb, eltArray.loadElement(data, capacity, idx)))
  }

  def swap(cb: EmitCodeBuilder, p: Value[Int], q: Value[Int]): Unit = {
    val pOff = elementOffset(cb, p)
    val qOff = elementOffset(cb, q)
    val tmpOff = elementOffset(cb, size)
    cb += Region.copyFrom(pOff, tmpOff, eltType.byteSize)
    cb += Region.copyFrom(qOff, pOff, eltType.byteSize)
    cb += Region.copyFrom(tmpOff, qOff, eltType.byteSize)
  }

  private def resize(cb: EmitCodeBuilder): Unit = {
    val newDataOffset = kb.genFieldThisRef[Long]("new_data_offset")
    cb.if_(size.ceq(capacity),
      {
        cb.assign(capacity, capacity * 2)
        cb.assign(data, eltArray.padWithMissing(cb, region, size, capacity, data))
      })
  }
}
