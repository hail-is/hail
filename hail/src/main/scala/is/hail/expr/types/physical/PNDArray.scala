package is.hail.expr.types.physical

import is.hail.annotations.{CodeOrdering, Region, StagedRegionValueBuilder, UnsafeOrdering}
import is.hail.asm4s.{ClassFieldRef, Code, MethodBuilder, _}
import is.hail.expr.Nat
import is.hail.expr.ir.{EmitMethodBuilder, coerce}
import is.hail.expr.types.virtual.TNDArray
import is.hail.utils._


final case class PNDArray(elementType: PType, nDims: Int, override val required: Boolean = false) extends PType {
  lazy val virtualType: TNDArray = TNDArray(elementType.virtualType, Nat(nDims), required)
  assert(elementType.required, "elementType must be required")

  override def _toPretty = s"NDArray[$elementType,$nDims]"

  override def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = throw new UnsupportedOperationException

  val flags = new StaticallyKnownField(PInt32Required, (r, off) => Region.loadInt(representation.loadField(r, off, "flags")))
  val offset = new StaticallyKnownField(
    PInt32Required,
    (r, off) => Region.loadInt(representation.loadField(r, off, "offset"))
  )
  val shape = new StaticallyKnownField(
    PTuple(true, Array.tabulate(nDims)(_ => PInt64Required):_*),
    (r, off) => representation.loadField(r, off, "shape")
  )
  val strides = new StaticallyKnownField(
    PTuple(true, Array.tabulate(nDims)(_ => PInt64Required):_*),
    (r, off) => representation.loadField(r, off, "strides")
  )

  val data = new StaticallyKnownField(
    PArray(elementType, required = true),
    (r, off) => representation.loadField(r, off, "data")
  )

  val representation: PStruct = {
    PStruct(required,
      ("flags", flags.pType),
      ("offset", offset.pType),
      ("shape", shape.pType),
      ("strides", strides.pType),
      ("data", data.pType))
  }

  override def byteSize: Long = representation.byteSize

  override def alignment: Long = representation.alignment

  override def unsafeOrdering(): UnsafeOrdering = representation.unsafeOrdering()

  override def fundamentalType: PType = representation.fundamentalType

  def numElements(shape: Code[Long], mb: MethodBuilder): Code[Long] = {
    def getShapeAtIdx(idx: Int): Code[Long] = Region.loadLong(this.representation.fieldType("shape").asInstanceOf[PTuple]
      .loadField(shape, idx))

      Array.range(0, nDims).foldLeft(const(1L)) { (prod, idx) => prod * getShapeAtIdx(idx) }
  }

  def makeDefaultStrides(sourceShapePType: PTuple, sourceShape: Code[Long], mb: MethodBuilder): Code[Long] = {
    def getShapeAtIdx(index: Int) = Region.loadLong(sourceShapePType.loadField(sourceShape, index))

    val stridesPType = this.representation.fieldType("strides").asInstanceOf[PTuple]
    val tupleStartAddress = mb.newField[Long]
    val runningProduct = mb.newLocal[Long]
    val region = mb.getArg[Region](1)

    Code(
      tupleStartAddress := stridesPType.allocate(region),
      runningProduct := elementType.byteSize,
      Code.foreach((nDims - 1) to 0 by -1) { idx =>
        val fieldOffset = stridesPType.fieldOffset(tupleStartAddress, idx)
        Code(
          Region.storeLong(fieldOffset, runningProduct),
          runningProduct := runningProduct * getShapeAtIdx(idx))
      },
      tupleStartAddress
    )
  }

  def getElementPosition(indices: Seq[Settable[Long]], nd: Code[Long], region: Code[Region], mb: MethodBuilder): Code[Long] = {
    val rep = this.representation
    val strides = rep.loadField(region, nd, "strides")
    val dataCode = rep.loadField(region, nd, "data")
    val dataP = rep.fieldType("data").asInstanceOf[PArray]
    def getStrideAtIdx(idx: Int): Code[Long] = Region.loadLong(rep.fieldType("strides").asInstanceOf[PTuple].loadField(strides, idx))
    val bytesAway = mb.newLocal[Long]
    val data = mb.newLocal[Long]
    coerce[Long](Code(
      data := dataCode,
      bytesAway := 0L,
      indices.zipWithIndex.foldLeft(Code._empty[Unit]){case (codeSoFar: Code[_], (requestedIndex: Settable[Long], strideIndex: Int)) =>
        Code(
          codeSoFar,
          bytesAway := bytesAway + requestedIndex * getStrideAtIdx(strideIndex))
      },
      bytesAway + dataP.elementOffset(data, dataP.loadLength(data), 0)
    ))
  }

  def construct(flags: Code[Int], offset: Code[Int], shape: Code[Long], strides: Code[Long], data: Code[Long], mb: MethodBuilder): Code[Long] = {
    val srvb = new StagedRegionValueBuilder(mb, this.representation)
    val shapeP = this.representation.fieldType("shape").asInstanceOf[PTuple]

    coerce[Long](Code(
      srvb.start(),
      srvb.addInt(flags),
      srvb.advance(),
      srvb.addInt(offset),
      srvb.advance(),
      srvb.addIRIntermediate(this.representation.fieldType("shape"))(shape),
      srvb.advance(),
      srvb.addIRIntermediate(this.representation.fieldType("strides"))(this.makeDefaultStrides(shapeP, shape, mb)),
      srvb.advance(),
      srvb.addIRIntermediate(this.representation.fieldType("data"))(data),
      srvb.end()
    ))
  }
}
