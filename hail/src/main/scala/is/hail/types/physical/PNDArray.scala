package is.hail.types.physical

import is.hail.annotations.Region
import is.hail.asm4s.{Code, _}
import is.hail.expr.Nat
import is.hail.expr.ir.orderings.CodeOrdering
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.stypes.SCode
import is.hail.types.physical.stypes.concrete.SNDArrayPointerCode
import is.hail.types.physical.stypes.interfaces.{SIndexableCode, SNDArrayCode, SNDArrayValue}
import is.hail.types.virtual.TNDArray

object PNDArray {
  val headerBytes = 16L
  def getReferenceCount(ndAddr: Long): Long = Region.loadLong(ndAddr - 16L)
  def storeReferenceCount(ndAddr: Long, newCount: Long): Unit = Region.storeLong(ndAddr - 16L, newCount)
  def getByteSize(ndAddr: Long): Long = Region.loadLong(ndAddr - 8L)
  def storeByteSize(ndAddr: Long, byteSize: Long): Unit = Region.storeLong(ndAddr - 8L, byteSize)
}

abstract class PNDArray extends PType {
  val elementType: PType
  val nDims: Int

  assert(elementType.isRealizable)

  lazy val virtualType: TNDArray = TNDArray(elementType.virtualType, Nat(nDims))
  assert(elementType.required, "elementType must be required")

  def dataFirstElementPointer(ndAddr: Code[Long]): Code[Long]
  def dataPArrayPointer(ndAddr: Code[Long]): Code[Long]

  def loadShape(off: Long, idx: Int): Long
  def unstagedLoadShapes(addr: Long): IndexedSeq[Long] = {
    (0 until nDims).map { dimIdx =>
      this.loadShape(addr, dimIdx)
    }
  }

  def loadShapes(cb: EmitCodeBuilder, addr: Value[Long], settables: IndexedSeq[Settable[Long]]): Unit
  def loadStrides(cb: EmitCodeBuilder, addr: Value[Long], settables: IndexedSeq[Settable[Long]]): Unit
  def unstagedLoadStrides(addr: Long): IndexedSeq[Long]

  def numElements(shape: IndexedSeq[Value[Long]]): Code[Long]
  
  def makeRowMajorStrides(sourceShapeArray: IndexedSeq[Value[Long]], region: Value[Region], cb: EmitCodeBuilder): IndexedSeq[Value[Long]]

  def makeColumnMajorStrides(sourceShapeArray: IndexedSeq[Value[Long]], region: Value[Region], cb: EmitCodeBuilder): IndexedSeq[Value[Long]]

  def getElementAddress(indices: IndexedSeq[Long], nd: Long): Long

  def loadElement(cb: EmitCodeBuilder, indices: IndexedSeq[Value[Long]], ndAddress: Value[Long]): SCode

  def constructByCopyingArray(
    shape: IndexedSeq[Value[Long]],
    strides: IndexedSeq[Value[Long]],
    data: SIndexableCode,
    cb: EmitCodeBuilder,
    region: Value[Region]
  ): PNDArrayCode

  def constructDataFunction(
    shape: IndexedSeq[Value[Long]],
    strides: IndexedSeq[Value[Long]],
    cb: EmitCodeBuilder,
    region: Value[Region]
  ): (Value[Long], EmitCodeBuilder =>  SNDArrayPointerCode)
}

abstract class PNDArrayValue extends PValue with SNDArrayValue {
  def pt: PNDArray
}

abstract class PNDArrayCode extends PCode with SNDArrayCode {
  def pt: PNDArray

  def memoize(cb: EmitCodeBuilder, name: String): PNDArrayValue
}
