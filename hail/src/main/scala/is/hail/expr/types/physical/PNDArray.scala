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

  lazy val virtualType: TNDArray = TNDArray(elementType.virtualType, Nat(nDims))
  assert(elementType.required, "elementType must be required")

  def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = throw new UnsupportedOperationException

  val flags: StaticallyKnownField[PInt32Required.type, Int]
  val offset: StaticallyKnownField[PInt32Required.type, Int]
  val shape: StaticallyKnownField[PTuple, Long]
  val strides: StaticallyKnownField[PTuple, Long]
  val data: StaticallyKnownField[PArray, Long]

  val representation: PStruct

  def dimensionLength(off: Code[Long], idx: Int): Code[Long] = {
    Region.loadLong(shape.pType.fieldOffset(shape.load(off), idx))
  }

  def numElements(shape: IndexedSeq[Code[Long]], mb: MethodBuilder): Code[Long]

  def makeShapeBuilder(shapeArray: IndexedSeq[Code[Long]]): StagedRegionValueBuilder => Code[Unit]

  def makeDefaultStridesBuilder(sourceShapeArray: IndexedSeq[Code[Long]], mb: MethodBuilder): StagedRegionValueBuilder => Code[Unit]

  def loadElementToIRIntermediate(indices: IndexedSeq[Code[Long]], ndAddress: Code[Long], mb: MethodBuilder): Code[_]

  def outOfBounds(indices: IndexedSeq[Code[Long]], nd: Code[Long], mb: MethodBuilder): Code[Boolean]

  def linearizeIndicesRowMajor(indices: IndexedSeq[Code[Long]], shapeArray: IndexedSeq[Code[Long]], mb: MethodBuilder): Code[Long]

  def unlinearizeIndexRowMajor(index: Code[Long], shapeArray: IndexedSeq[Code[Long]], mb: MethodBuilder): (Code[Unit], IndexedSeq[Code[Long]])

  def copyRowMajorToColumnMajor(rowMajorAddress: Code[Long], targetAddress: Code[Long], nRows: Code[Long], nCols: Code[Long], mb: MethodBuilder): Code[Unit]

  def copyColumnMajorToRowMajor(colMajorAddress: Code[Long], targetAddress: Code[Long], nRows: Code[Long], nCols: Code[Long], mb: MethodBuilder): Code[Unit]

  def construct(flags: Code[Int], offset: Code[Int], shapeBuilder: (StagedRegionValueBuilder => Code[Unit]),
    stridesBuilder: (StagedRegionValueBuilder => Code[Unit]), data: Code[Long], mb: MethodBuilder): Code[Long]
}
