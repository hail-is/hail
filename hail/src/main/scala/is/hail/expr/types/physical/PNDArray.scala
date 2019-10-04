package is.hail.expr.types.physical

import is.hail.annotations.{CodeOrdering, Region, StagedRegionValueBuilder, UnsafeOrdering}
import is.hail.asm4s.{Code, MethodBuilder, _}
import is.hail.expr.Nat
import is.hail.expr.ir.{EmitMethodBuilder}
import is.hail.expr.types.virtual.TNDArray
import is.hail.utils._
import is.hail.asm4s._


final case class PNDArray(elementType: PType, nDims: Int, override val required: Boolean = false) extends PType {
  lazy val virtualType: TNDArray = TNDArray(elementType.virtualType, Nat(nDims), required)
  assert(elementType.required, "elementType must be required")

  def _asIdent: String = s"ndarray_of_${elementType.asIdent}"
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

  def numElements(shape: Array[Code[Long]], mb: MethodBuilder): Code[Long] = {
      Array.range(0, nDims).foldLeft(const(1L)) { (prod, idx) => prod * shape(idx) }
  }

  def makeDefaultStrides(sourceShapeArray: Array[Code[Long]], mb: MethodBuilder): Code[Long] = {
    val tupleStartAddress = mb.newField[Long]
    val runningProduct = mb.newLocal[Long]
    val region = mb.getArg[Region](1)
    val tempShapeStorage = mb.newLocal[Long]

    Code(
      tupleStartAddress := strides.pType.allocate(region),
      runningProduct := elementType.byteSize,
      Code.foreach((nDims - 1) to 0 by -1) { idx =>
        val fieldOffset = strides.pType.fieldOffset(tupleStartAddress, idx)
        Code(
          Region.storeLong(fieldOffset, runningProduct),
          tempShapeStorage := sourceShapeArray(idx),
          runningProduct := runningProduct * (tempShapeStorage > 0L).mux(tempShapeStorage, 1L))
      },
      tupleStartAddress
    )
  }

  def getElementPosition(indices: Array[Code[Long]], nd: Code[Long], region: Code[Region], mb: MethodBuilder): Code[Long] = {
    val stridesTuple  = new CodePTuple(strides.pType, region, strides.load(region, nd))
    val bytesAway = mb.newLocal[Long]
    val dataStore = mb.newLocal[Long]

    coerce[Long](Code(
      dataStore := data.load(region, nd),
      bytesAway := 0L,
      indices.zipWithIndex.foldLeft(Code._empty[Unit]){case (codeSoFar: Code[_], (requestedIndex: Code[Long], strideIndex: Int)) =>
        Code(
          codeSoFar,
          bytesAway := bytesAway + requestedIndex * stridesTuple(strideIndex))
      },
      bytesAway + data.pType.elementOffset(dataStore, data.pType.loadLength(dataStore), 0)
    ))
  }

  def construct(flags: Code[Int], offset: Code[Int], shape: Code[Long], strides: Code[Long], data: Code[Long], mb: MethodBuilder): Code[Long] = {
    val srvb = new StagedRegionValueBuilder(mb, this.representation)

    coerce[Long](Code(
      srvb.start(),
      srvb.addInt(flags),
      srvb.advance(),
      srvb.addInt(offset),
      srvb.advance(),
      srvb.addIRIntermediate(this.representation.fieldType("shape"))(shape),
      srvb.advance(),
      srvb.addIRIntermediate(this.representation.fieldType("strides"))(strides),
      srvb.advance(),
      srvb.addIRIntermediate(this.representation.fieldType("data"))(data),
      srvb.end()
    ))
  }

  def construct2(flags: Code[Int], offset: Code[Int], shapeBuilder: (StagedRegionValueBuilder => Code[Unit]), strides: Code[Long], data: Code[Long], mb: MethodBuilder): Code[Long] = {
    val srvb = new StagedRegionValueBuilder(mb, this.representation)

    coerce[Long](Code(
      srvb.start(),
      srvb.addInt(flags),
      srvb.advance(),
      srvb.addInt(offset),
      srvb.advance(),
      srvb.addBaseStruct(this.shape.pType, shapeBuilder),
      srvb.advance(),
      srvb.addIRIntermediate(this.representation.fieldType("strides"))(strides),
      srvb.advance(),
      srvb.addIRIntermediate(this.representation.fieldType("data"))(data),
      srvb.end()
    ))
  }
}
