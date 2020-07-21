package is.hail.types.encoded
import java.util.UUID

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.functions.StringFunctions
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, EmitRegion, typeToTypeInfo}
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.{PCanonicalArray, PCanonicalNDArray, PCanonicalNDArraySettable, PNDArrayValue, PType}
import is.hail.types.virtual.{TNDArray, Type}
import is.hail.utils._

case class ENDArrayColumnMajor(elementType: EType, nDims: Int, required: Boolean = false) extends EContainer {
  type DecodedPType = PCanonicalNDArray

  override def decodeCompatible(pt: PType): Boolean = {
    pt.required == required &&
      pt.isInstanceOf[DecodedPType] &&
      elementType.decodeCompatible(pt.asInstanceOf[DecodedPType].elementType)
  }

  override def encodeCompatible(pt: PType): Boolean = {
    pt.required == required &&
      pt.isInstanceOf[DecodedPType] &&
      elementType.encodeCompatible(pt.asInstanceOf[DecodedPType].elementType)
  }

  override def _buildEncoder(pt: PType, mb: EmitMethodBuilder[_], v: Value[_], out: Value[OutputBuffer]): Code[Unit] = {
    val uuid = UUID.randomUUID().toString
    val pnd = pt.asInstanceOf[PCanonicalNDArray]
    assert(pnd.elementType.required)
    val ndarray = coerce[Long](v)

    val firstShape = mb.newLocal[Long]("first_shape")

    val writeShapes = (0 until nDims).map(i => out.writeLong(pnd.loadShape(ndarray, i)))
    // Note, encoded strides is in terms of indices into ndarray, not bytes.
    val writeStrides = (0 until nDims).map(i => out.writeLong(pnd.loadStride(ndarray, i) / pnd.elementType.byteSize))

    val writeElemF = elementType.buildEncoder(pnd.elementType, mb.ecb)

    val idxVars = (0 until nDims).map(_ => mb.newLocal[Long]())

    val loadAndWrite = {
      //Code(
        //Code._println(const(s"$uuid: ENDArrayColumnMajor element write out. firstShape = ").concat(firstShape.toS)),
        writeElemF(pnd.loadElementToIRIntermediate(idxVars, ndarray, mb), out)
      //)
    }

    val columnMajorLoops = idxVars.zipWithIndex.foldLeft(loadAndWrite) { case (innerLoops, (dimVar, dimIdx)) =>
      Code(
        dimVar := 0L,
        Code.whileLoop(dimVar < pnd.loadShape(ndarray, dimIdx),
          innerLoops,
          dimVar := dimVar + 1L
        )
      )
    }

    Code(
      firstShape := pnd.loadShape(ndarray, 0),
      Code(writeShapes),
      //Code._println(const(s"$uuid: Starting columnMajorLoops. firstShape = ").concat(firstShape.toS)),
      columnMajorLoops
      //Code._println(const(s"$uuid: Finishing columnMajorLoops. firstShape = ").concat(firstShape.toS))
    )

//    EmitCodeBuilder.scopedVoid(mb){cb =>
//      cb.append(firstShape := pnd.loadShape(ndarray, 0))
//      cb.append(Code(writeShapes))
//      cb.append(Code._println(const(s"$uuid: Starting columnMajorLoops. firstShape = ").concat(firstShape.toS)))
//      cb.append(columnMajorLoops)
//      cb.append(Code._println(const(s"$uuid: Finishing columnMajorLoops. firstShape = ").concat(firstShape.toS)))
//    }
  }

  override def _buildDecoder(pt: PType, mb: EmitMethodBuilder[_], region: Value[Region], in: Value[InputBuffer]): Code[_] = {
    val pnd = pt.asInstanceOf[PCanonicalNDArray]
    val shapeVars = (0 until nDims).map(i => mb.newLocal[Long](s"ndarray_decoder_shape_$i"))
    val totalNumElements = mb.newLocal[Long]("ndarray_decoder_total_num_elements")

    val readElemF = elementType.buildInplaceDecoder(pnd.elementType, mb.ecb)
    val dataAddress = mb.newLocal[Long]("ndarray_decoder_data_addr")

    val dataIdx = mb.newLocal[Int]("ndarray_decoder_data_idx")

    val answer = mb.newLocal[Long]("answer_nd_decode")

    Code(
      totalNumElements := 1L,
      Code(shapeVars.map(shapeVar => Code(
        shapeVar := in.readLong(),
        totalNumElements := totalNumElements * shapeVar
      ))),
      //Code._println(const("Reading, shapeVars(0) = ").concat(shapeVars(0).toS).concat(", totalNumElmenets = ").concat(totalNumElements.toS)),
      dataAddress := pnd.data.pType.allocate(region, totalNumElements.toI),
      pnd.data.pType.stagedInitialize(dataAddress, totalNumElements.toI),
      Code.forLoop(dataIdx := 0, dataIdx < totalNumElements.toI, dataIdx := dataIdx + 1,
        readElemF(region, pnd.data.pType.elementOffset(dataAddress, totalNumElements.toI, dataIdx), in)
      ),
      {
        val er = EmitRegion(mb, region)
        Code._println(StringFunctions.boxArg(er, pnd.data.pType)(dataAddress))
      },
      answer := pnd.construct(pnd.makeShapeBuilder(shapeVars), pnd.makeColumnMajorStridesBuilder(shapeVars, mb), dataAddress, mb),
      {
        val er = EmitRegion(mb, region)
        Code._println(StringFunctions.boxArg(er, pnd)(answer))
      },
      answer
    )
  }

  override def _buildSkip(mb: EmitMethodBuilder[_], r: Value[Region], in: Value[InputBuffer]): Code[Unit] = {
    val totalNumElements = mb.newLocal[Long]("ndarray_skipper_total_num_elements")
    val dataIdx = mb.newLocal[Int]("ndarray_skipper_data_idx")
    val skip = elementType.buildSkip(mb)

    Code(
      totalNumElements := 1L,
      Code((0 until nDims).map { _ =>
        totalNumElements := totalNumElements * in.readLong()
      }),
      Code.forLoop(dataIdx := 0, dataIdx < totalNumElements.toI, dataIdx := dataIdx + 1,
        skip(r, in)
      )
    )
  }

  def _decodedPType(requestedType: Type): PType = {
    val requestedTNDArray = requestedType.asInstanceOf[TNDArray]
    val elementPType = elementType.decodedPType(requestedTNDArray.elementType)
    PCanonicalNDArray(elementPType, requestedTNDArray.nDims, required)
  }
  override def setRequired(required: Boolean): EType = ENDArrayColumnMajor(elementType, nDims, required)

  override def _asIdent = s"ndarray_of_${elementType.asIdent}"
  override def _toPretty = s"ENDArrayColumnMajor[$elementType,$nDims]"
}
