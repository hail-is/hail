package is.hail.types.physical

import is.hail.annotations.{CodeOrdering, Region, StagedRegionValueBuilder}
import is.hail.asm4s.{Code, _}
import is.hail.expr.Nat
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.stypes.interfaces.{SNDArrayCode, SNDArrayValue}
import is.hail.types.virtual.TNDArray

abstract class StaticallyKnownField[T, U](val pType: T) {
  def load(off: Code[Long])(implicit line: LineNumber): Code[U]
}

abstract class StagedRegionValue {
  def apply(srvb: StagedRegionValueBuilder)(implicit line: LineNumber): Code[Unit]
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

  def loadShape(cb: EmitCodeBuilder, off: Code[Long], idx: Int)(implicit line: LineNumber): Code[Long]

  def loadStride(cb: EmitCodeBuilder, off: Code[Long], idx: Int)(implicit line: LineNumber): Code[Long]

  def numElements(shape: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_])(implicit line: LineNumber): Code[Long]

  def makeShapeBuilder(shapeArray: IndexedSeq[Value[Long]])(implicit line: LineNumber): StagedRegionValueBuilder => Code[Unit]

  def makeRowMajorStridesBuilder(sourceShapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_])(implicit line: LineNumber): StagedRegionValueBuilder => Code[Unit]

  def makeColumnMajorStridesBuilder(sourceShapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_])(implicit line: LineNumber): StagedRegionValueBuilder => Code[Unit]

  def setElement(indices: IndexedSeq[Value[Long]], ndAddress: Value[Long], newElement: Code[_], mb: EmitMethodBuilder[_])(implicit line: LineNumber): Code[Unit]

  def loadElement(cb: EmitCodeBuilder, indices: IndexedSeq[Value[Long]], ndAddress: Value[Long])(implicit line: LineNumber): Code[Long]
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
  ): PNDArrayCode
}

abstract class PNDArrayValue extends PValue with SNDArrayValue {
  def pt: PNDArray
}

abstract class PNDArrayCode extends PCode with SNDArrayCode {
  def pt: PNDArray

  def memoize(cb: EmitCodeBuilder, name: String): PNDArrayValue
}
