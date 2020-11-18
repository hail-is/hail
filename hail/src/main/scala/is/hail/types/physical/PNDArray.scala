package is.hail.types.physical

import is.hail.annotations.{CodeOrdering, Region, StagedRegionValueBuilder}
import is.hail.asm4s.{Code, _}
import is.hail.expr.Nat
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.virtual.TNDArray

final class StaticallyKnownField[T, U](
  val pType: T,
  val load: Code[Long] => Code[U]
)

abstract class PNDArray extends PType {
  val elementType: PType
  val nDims: Int

  assert(elementType.isRealizable)

  lazy val virtualType: TNDArray = TNDArray(elementType.virtualType, Nat(nDims))
  assert(elementType.required, "elementType must be required")

  def codeOrdering(mb: EmitMethodBuilder[_], other: PType): CodeOrdering = throw new UnsupportedOperationException

  val shape: StaticallyKnownField[PTuple, Long]
  val strides: StaticallyKnownField[PTuple, Long]
  val data: StaticallyKnownField[PArray, Long]

  val representation: PStruct

  def dimensionLength(off: Code[Long], idx: Int): Code[Long] = {
    Region.loadLong(shape.pType.fieldOffset(shape.load(off), idx))
  }

  def loadShape(cb: EmitCodeBuilder, off: Code[Long], idx: Int): Code[Long]

  def loadStride(cb: EmitCodeBuilder, off: Code[Long], idx: Int): Code[Long]

  def numElements(shape: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): Code[Long]

  def makeShapeBuilder(shapeArray: IndexedSeq[Value[Long]]): StagedRegionValueBuilder => Code[Unit]

  def makeRowMajorStridesBuilder(sourceShapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): StagedRegionValueBuilder => Code[Unit]

  def makeColumnMajorStridesBuilder(sourceShapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): StagedRegionValueBuilder => Code[Unit]

  def setElement(indices: IndexedSeq[Value[Long]], ndAddress: Value[Long], newElement: Code[_], mb: EmitMethodBuilder[_]): Code[Unit]

  def loadElement(cb: EmitCodeBuilder, indices: IndexedSeq[Value[Long]], ndAddress: Value[Long]): Code[Long]
  def loadElementToIRIntermediate(indices: IndexedSeq[Value[Long]], ndAddress: Value[Long], mb: EmitMethodBuilder[_]): Code[_]

  def linearizeIndicesRowMajor(indices: IndexedSeq[Code[Long]], shapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): Code[Long]

  def unlinearizeIndexRowMajor(index: Code[Long], shapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): (Code[Unit], IndexedSeq[Value[Long]])

  def construct(
    shapeBuilder: StagedRegionValueBuilder => Code[Unit],
    stridesBuilder: StagedRegionValueBuilder => Code[Unit],
    data: Code[Long],
    mb: EmitMethodBuilder[_],
    region: Value[Region]
  ): PNDArrayCode
}

abstract class PNDArrayValue extends PValue {
  def loadElement(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): PCode

  def shapes(cb: EmitCodeBuilder): IndexedSeq[Value[Long]]

  def strides(cb: EmitCodeBuilder): IndexedSeq[Value[Long]]

  def pt: PNDArray

  def outOfBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder): Code[Boolean]

  def assertInBounds(indices: IndexedSeq[Value[Long]], cb: EmitCodeBuilder, errorId: Int = -1): Code[Unit]

  def sameShape(other: PNDArrayValue, cb: EmitCodeBuilder): Code[Boolean]
}

abstract class PNDArrayCode extends PCode {
  def pt: PNDArray

  def shape: PBaseStructCode

  def memoize(cb: EmitCodeBuilder, name: String): PNDArrayValue
}
