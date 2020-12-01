package is.hail.types.physical

import is.hail.annotations.{CodeOrdering, Region, StagedRegionValueBuilder}
import is.hail.asm4s.{Code, _}
import is.hail.expr.Nat
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.virtual.TNDArray

abstract class StaticallyKnownField[T, U](val pType: T) {
  def load(off: Code[Long])(implicit line: LineNumber): Code[U]
}

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

  def dimensionLength(off: Code[Long], idx: Int)(implicit line: LineNumber): Code[Long] = {
    Region.loadLong(shape.pType.fieldOffset(shape.load(off), idx))
  }

  def numElements(shape: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_])(implicit line: LineNumber): Code[Long]

  def makeShapeBuilder(shapeArray: IndexedSeq[Value[Long]])(implicit line: LineNumber): StagedRegionValueBuilder => Code[Unit]

  def makeRowMajorStridesBuilder(sourceShapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_])(implicit line: LineNumber): StagedRegionValueBuilder => Code[Unit]

  def makeColumnMajorStridesBuilder(sourceShapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_])(implicit line: LineNumber): StagedRegionValueBuilder => Code[Unit]

  def setElement(indices: IndexedSeq[Value[Long]], ndAddress: Value[Long], newElement: Code[_], mb: EmitMethodBuilder[_])(implicit line: LineNumber): Code[Unit]

  def loadElementToIRIntermediate(indices: IndexedSeq[Value[Long]], ndAddress: Value[Long], mb: EmitMethodBuilder[_])(implicit line: LineNumber): Code[_]

  def linearizeIndicesRowMajor(indices: IndexedSeq[Code[Long]], shapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_])(implicit line: LineNumber): Code[Long]

  def unlinearizeIndexRowMajor(index: Code[Long], shapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_])(implicit line: LineNumber): (Code[Unit], IndexedSeq[Value[Long]])

  def construct(
    shapeBuilder: StagedRegionValueBuilder => Code[Unit],
    stridesBuilder: StagedRegionValueBuilder => Code[Unit],
    data: Code[Long],
    mb: EmitMethodBuilder[_],
    region: Value[Region]
  )(implicit line: LineNumber
  ): Code[Long]
}

abstract class PNDArrayValue extends PValue {
  def apply(indices: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_])(implicit line: LineNumber): Value[_]

  def shapes(): IndexedSeq[Value[Long]]

  def strides(): IndexedSeq[Value[Long]]

  override def pt: PNDArray = ???

  def outOfBounds(indices: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_])(implicit line: LineNumber): Code[Boolean]

  def assertInBounds(indices: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_], errorId: Int = -1)(implicit line: LineNumber): Code[Unit]

  def sameShape(other: PNDArrayValue, mb: EmitMethodBuilder[_])(implicit line: LineNumber): Code[Boolean]
}

abstract class PNDArrayCode extends PCode {
  override def pt: PNDArray

  def shape(implicit line: LineNumber): PBaseStructCode

  def memoize(cb: EmitCodeBuilder, name: String)(implicit line: LineNumber): PNDArrayValue
}
