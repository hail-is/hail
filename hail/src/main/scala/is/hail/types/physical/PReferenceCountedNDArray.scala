package is.hail.types.physical
import is.hail.annotations.{Region, StagedRegionValueBuilder, UnsafeOrdering}
import is.hail.asm4s.{Code, Value}
import is.hail.expr.ir.EmitMethodBuilder

class PReferenceCountedNDArray(elementType: PType, nDims: Int, required: Boolean = false) extends PNDArray {
  override val shape: StaticallyKnownField[PTuple, Long] = _
  override val strides: StaticallyKnownField[PTuple, Long] = _
  override val data: StaticallyKnownField[PArray, Long] = _
  override val representation: PStruct = _

  override def numElements(shape: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): Code[Long] = ???

  override def makeShapeBuilder(shapeArray: IndexedSeq[Value[Long]]): StagedRegionValueBuilder => Code[Unit] = ???

  override def makeRowMajorStridesBuilder(sourceShapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): StagedRegionValueBuilder => Code[Unit] = ???

  override def makeColumnMajorStridesBuilder(sourceShapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): StagedRegionValueBuilder => Code[Unit] = ???

  override def setElement(indices: IndexedSeq[Value[Long]], ndAddress: Value[Long], newElement: Code[_], mb: EmitMethodBuilder[_]): Code[Unit] = ???

  override def loadElementToIRIntermediate(indices: IndexedSeq[Value[Long]], ndAddress: Value[Long], mb: EmitMethodBuilder[_]): Code[_] = ???

  override def linearizeIndicesRowMajor(indices: IndexedSeq[Code[Long]], shapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): Code[Long] = ???

  override def unlinearizeIndexRowMajor(index: Code[Long], shapeArray: IndexedSeq[Value[Long]], mb: EmitMethodBuilder[_]): (Code[Unit], IndexedSeq[Value[Long]]) = ???

  override def construct(shapeBuilder: StagedRegionValueBuilder => Code[Unit], stridesBuilder: StagedRegionValueBuilder => Code[Unit], data: Code[Long], mb: EmitMethodBuilder[_], region: Value[Region]): Code[Long] = ???

  override def unsafeOrdering(): UnsafeOrdering = ???

  override def _asIdent: String = ???

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = ???

  override def encodableType: PType = ???

  override def setRequired(required: Boolean): PType = ???

  override def containsPointers: Boolean = ???

  override def copyFromType(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Long] = ???

  override def copyFromTypeAndStackValue(mb: EmitMethodBuilder[_], region: Value[Region], srcPType: PType, stackValue: Code[_], deepCopy: Boolean): Code[_] = ???

  override protected def _copyFromAddress(region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Long = ???

  override def constructAtAddress(mb: EmitMethodBuilder[_], addr: Code[Long], region: Value[Region], srcPType: PType, srcAddress: Code[Long], deepCopy: Boolean): Code[Unit] = ???

  override def constructAtAddress(addr: Long, region: Region, srcPType: PType, srcAddress: Long, deepCopy: Boolean): Unit = ???

  override def required: Boolean = ???
}
