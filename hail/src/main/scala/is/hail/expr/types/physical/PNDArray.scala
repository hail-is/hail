package is.hail.expr.types.physical

import is.hail.annotations.{CodeOrdering, Region, StagedRegionValueBuilder, UnsafeOrdering}
import is.hail.asm4s.{ClassFieldRef, Code, MethodBuilder, _}
import is.hail.expr.Nat
import is.hail.expr.ir.EmitMethodBuilder
import is.hail.expr.types.virtual.TNDArray

final case class PNDArray(elementType: PType, nDims: Int, override val required: Boolean = false) extends PType {
  lazy val virtualType: TNDArray = TNDArray(elementType.virtualType, Nat(nDims), required)
  assert(elementType.required, "elementType must be required")

  override def _toPretty = s"NDArray[$elementType,$nDims]"

  override def codeOrdering(mb: EmitMethodBuilder, other: PType): CodeOrdering = throw new UnsupportedOperationException

  val representation: PStruct = {
    PStruct(required,
      ("flags", PInt32Required),
      ("offset", PInt32Required),
      ("shape", PTuple(true, Array.tabulate(nDims)(_ => PInt64Required):_*)),
      ("strides", PTuple(true, Array.tabulate(nDims)(_ => PInt64Required):_*)),
      ("data", PArray(elementType, required = true)))
  }

  override def byteSize: Long = representation.byteSize

  override def alignment: Long = representation.alignment

  override def unsafeOrdering(): UnsafeOrdering = representation.unsafeOrdering()

  override def fundamentalType: PType = representation.fundamentalType

  def numElements(shape: Code[Long], mb: MethodBuilder): Code[Long] = {
    def getShapeAtIdx(idx: Int): Code[Long] = Region.loadLong(this.representation.fieldType("shape").asInstanceOf[PTuple]
      .loadField(shape, idx))

    if (nDims == 0) {
      0L
    }
    else {
      val totalElements = mb.newField[Long]
      Code(
        totalElements := 1L,
        Code.foreach(0 until nDims) { index =>
          totalElements := totalElements * getShapeAtIdx(index)
        },
        totalElements
      )
    }
  }

  def makeDefaultStrides(getShapeAtIdx: (Int) => Code[Long], srvb: StagedRegionValueBuilder, mb: MethodBuilder): Code[Long] = {
    val stridesPType = this.representation.fieldType("strides").asInstanceOf[PTuple]
    val tupleStartAddress = mb.newField[Long]
    (Code (
      srvb.start(),
      tupleStartAddress := srvb.offset,
      // Fill with 0s, then backfill with actual data
      Code.foreach(0 until nDims) { index =>
        Code(srvb.addLong(0L), srvb.advance())
      },
      {
        val runningProduct = mb.newField[Long]
        Code(
          runningProduct := elementType.byteSize,
          Code.foreach((nDims - 1) to 0 by -1) { idx =>
            val fieldOffset = stridesPType.fieldOffset(tupleStartAddress, idx)
            Code(
              Region.storeLong(fieldOffset, runningProduct),
              runningProduct := runningProduct * getShapeAtIdx(idx)
            )
          }
        )
      }
    )).asInstanceOf[Code[Long]]
  }

  def getElementPosition(indices: Seq[ClassFieldRef[Long]], nd: Code[Long], region: Code[Region], mb: MethodBuilder): Code[Long] = {
    val rep = this.representation
    val strides = rep.loadField(region, nd, "strides")
    val data = rep.loadField(region, nd, "data")
    val dataLength = rep.fieldType("data").asInstanceOf[PArray].loadLength(data)
    def getStrideAtIdx(idx: Int): Code[Long] = Region.loadLong(rep.fieldType("strides").asInstanceOf[PTuple].loadField(strides, idx))
    val bytesAway = mb.newField[Long]
    val temp = mb.newField[Long]
    val currentStride = mb.newField[Long]
    coerce[Long](Code(
      bytesAway := 0L,
      currentStride := 0L,
      indices.zipWithIndex.foldLeft(Code._empty[Unit]){case (codeSoFar: Code[_], (requestedIndex: ClassFieldRef[Long], strideIndex: Int)) =>
        Code(
          codeSoFar,
          //currentStride := getStrideAtIdx(strideIndex),
          Code.getStatic[java.lang.System, java.io.PrintStream]("out").invoke[String, Unit](
            "println", "requestedIndex is:"),
          Code.getStatic[java.lang.System, java.io.PrintStream]("out").invoke[Long, Unit](
            "println", requestedIndex),
          Code.getStatic[java.lang.System, java.io.PrintStream]("out").invoke[String, Unit](
            "println", "currentStride is:"),
          Code.getStatic[java.lang.System, java.io.PrintStream]("out").invoke[Long, Unit](
            "println", currentStride),
          bytesAway := bytesAway + requestedIndex * getStrideAtIdx(strideIndex)
        )
      },

      temp := rep.fieldType("data").asInstanceOf[PArray].elementOffset(data, dataLength, 0),

      Code.getStatic[java.lang.System, java.io.PrintStream]("out").invoke[String, Unit](
        "println", "Computed bytesAway"),
      Code.getStatic[java.lang.System, java.io.PrintStream]("out").invoke[String, Unit](
        "println", "BytesAway is:"),
      Code.getStatic[java.lang.System, java.io.PrintStream]("out").invoke[Long, Unit](
        "println", bytesAway),
      Code.getStatic[java.lang.System, java.io.PrintStream]("out").invoke[String, Unit](
        "println", "Temp is:"),
      Code.getStatic[java.lang.System, java.io.PrintStream]("out").invoke[Long, Unit](
        "println", temp),
      bytesAway + temp
    ))
  }
}
