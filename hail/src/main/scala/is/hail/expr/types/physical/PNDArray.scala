package is.hail.expr.types.physical

import is.hail.annotations.{CodeOrdering, Region, StagedRegionValueBuilder}
import is.hail.asm4s.{Code, MethodBuilder, _}
import is.hail.expr.Nat
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TNDArray

final class StaticallyKnownField[T, U](
  val pType: T,
  val load: Code[Long] => Code[U]
)

object PNDArray {
  def apply(elementType: PType, nDims: Int, required: Boolean = false) = PCanonicalNDArray(elementType, nDims, required)
}

abstract class PNDArray extends PType {
  val elementType: PType
  val nDims: Int

  lazy val virtualType: TNDArray = TNDArray(elementType.virtualType, Nat(nDims), required)
  assert(elementType.required, "elementType must be required")

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = throw new UnsupportedOperationException

  val flags: StaticallyKnownField[PInt32Required.type, Int]
  val offset: StaticallyKnownField[PInt32Required.type, Int]
  val shape: StaticallyKnownField[PTuple, Long]
  val strides: StaticallyKnownField[PTuple, Long]
  val data: StaticallyKnownField[PArray, Long]

  val representation: PStruct

  def copy(elementType: PType = this.elementType, nDims: Int = this.nDims, required: Boolean = this.required): PNDArray

  def dimensionLength(off: Code[Long], idx: Int): Code[Long] = {
    Region.loadLong(shape.pType.fieldOffset(shape.load(off), idx))
  }

  def numElements(shape: Array[Code[Long]], mb: MethodBuilder): Code[Long]

  def makeShapeBuilder(shapeArray: Array[Code[Long]]): StagedRegionValueBuilder => Code[Unit]

  def makeDefaultStridesBuilder(sourceShapeArray: Array[Code[Long]], mb: MethodBuilder): StagedRegionValueBuilder => Code[Unit]

  def getElementAddress(indices: Array[Code[Long]], nd: Code[Long], mb: MethodBuilder): Code[Long]

  def loadElementToIRIntermediate(indices: Array[Code[Long]], ndAddress: Code[Long], mb: MethodBuilder): Code[_]

  def outOfBounds(indices: Array[Code[Long]], nd: Code[Long], mb: MethodBuilder): Code[Boolean]

  def linearizeIndicesRowMajor(indices: Array[Code[Long]], shapeArray: Array[Code[Long]], mb: MethodBuilder): Code[Long]

  def unlinearizeIndexRowMajor(index: Code[Long], shapeArray: Array[Code[Long]], mb: MethodBuilder): (Code[Unit], Array[Code[Long]])

  def copyRowMajorToColumnMajor(rowMajorAddress: Code[Long], targetAddress: Code[Long], nRows: Code[Long], nCols: Code[Long], mb: MethodBuilder): Code[Unit]

  def copyColumnMajorToRowMajor(colMajorAddress: Code[Long], targetAddress: Code[Long], nRows: Code[Long], nCols: Code[Long], mb: MethodBuilder): Code[Unit]

  def construct(flags: Code[Int], offset: Code[Int], shapeBuilder: (StagedRegionValueBuilder => Code[Unit]),
    stridesBuilder: (StagedRegionValueBuilder => Code[Unit]), data: Code[Long], mb: MethodBuilder): Code[Long]
}
