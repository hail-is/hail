package is.hail.types.physical

import is.hail.annotations.{CodeOrdering, Region, StagedRegionValueBuilder}
import is.hail.asm4s.{Code, _}
import is.hail.expr.Nat
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder}
import is.hail.types.physical.stypes.interfaces.{SBaseStructCode, SNDArrayCode, SNDArrayValue}
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
  val data: StaticallyKnownField[PArray, Long]

  val representation: PStruct

  def dataFirstElementPointer(ndAddr: Code[Long]): Code[Long]
  def dataPArrayPointer(ndAddr: Code[Long]): Code[Long]

  def loadShape(cb: EmitCodeBuilder, off: Code[Long], idx: Int): Code[Long]

  def loadShape(off: Long, idx: Int): Long

  def loadStride(cb: EmitCodeBuilder, off: Code[Long], idx: Int): Code[Long]

  def numElements(shape: IndexedSeq[Value[Long]]): Code[Long]

  def makeShapeStruct(shapeArray: IndexedSeq[Value[Long]], region: Value[Region], cb: EmitCodeBuilder): SBaseStructCode

  def makeRowMajorStridesStruct(sourceShapeArray: IndexedSeq[Value[Long]], region: Value[Region], cb: EmitCodeBuilder): SBaseStructCode

  def makeColumnMajorStridesStruct(sourceShapeArray: IndexedSeq[Value[Long]], region: Value[Region], cb: EmitCodeBuilder): SBaseStructCode

  def getElementAddress(indices: IndexedSeq[Long], nd: Long): Long

  def loadElement(cb: EmitCodeBuilder, indices: IndexedSeq[Value[Long]], ndAddress: Value[Long]): Code[Long]
  def loadElementToIRIntermediate(indices: IndexedSeq[Value[Long]], ndAddress: Value[Long], cb: EmitCodeBuilder): Code[_]

  def construct(
    shapeCode: SBaseStructCode,
    stridesCode: SBaseStructCode,
    data: Code[Long],
    mb: EmitCodeBuilder,
    region: Value[Region]
  ): PNDArrayCode
}

abstract class PNDArrayValue extends PValue with SNDArrayValue {
  def pt: PNDArray
}

abstract class PNDArrayCode extends PCode with SNDArrayCode {
  def pt: PNDArray

  def memoize(cb: EmitCodeBuilder, name: String): PNDArrayValue
}
