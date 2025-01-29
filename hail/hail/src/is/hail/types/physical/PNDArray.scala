package is.hail.types.physical

import is.hail.annotations.Region
import is.hail.asm4s.{Code, _}
import is.hail.expr.Nat
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.types.physical.stypes.SValue
import is.hail.types.physical.stypes.concrete.SNDArrayPointerValue
import is.hail.types.physical.stypes.interfaces.{SIndexableValue, SNDArrayValue}
import is.hail.types.virtual.TNDArray

abstract class PNDArray extends PType {
  val elementType: PType
  val nDims: Int

  assert(elementType.isRealizable)

  lazy val virtualType: TNDArray = TNDArray(elementType.virtualType, Nat(nDims))
  assert(elementType.required, "elementType must be required")

  def dataFirstElementPointer(ndAddr: Code[Long]): Code[Long]

  def loadShape(off: Long, idx: Int): Long

  def unstagedLoadShapes(addr: Long): IndexedSeq[Long] =
    (0 until nDims).map(dimIdx => this.loadShape(addr, dimIdx))

  def loadShapes(cb: EmitCodeBuilder, addr: Value[Long], settables: IndexedSeq[Settable[Long]])
    : Unit

  def loadStrides(cb: EmitCodeBuilder, addr: Value[Long], settables: IndexedSeq[Settable[Long]])
    : Unit

  def unstagedLoadStrides(addr: Long): IndexedSeq[Long]

  def numElements(shape: IndexedSeq[Value[Long]]): Code[Long]

  def makeRowMajorStrides(sourceShapeArray: IndexedSeq[Value[Long]], cb: EmitCodeBuilder)
    : IndexedSeq[Value[Long]]

  def makeColumnMajorStrides(sourceShapeArray: IndexedSeq[Value[Long]], cb: EmitCodeBuilder)
    : IndexedSeq[Value[Long]]

  def getElementAddress(indices: IndexedSeq[Long], nd: Long): Long

  def loadElement(cb: EmitCodeBuilder, indices: IndexedSeq[Value[Long]], ndAddress: Value[Long])
    : SValue

  def constructByCopyingArray(
    shape: IndexedSeq[Value[Long]],
    strides: IndexedSeq[Value[Long]],
    data: SIndexableValue,
    cb: EmitCodeBuilder,
    region: Value[Region],
  ): SNDArrayValue

  def constructDataFunction(
    shape: IndexedSeq[Value[Long]],
    strides: IndexedSeq[Value[Long]],
    cb: EmitCodeBuilder,
    region: Value[Region],
  ): (Value[Long], EmitCodeBuilder => SNDArrayPointerValue)
}
