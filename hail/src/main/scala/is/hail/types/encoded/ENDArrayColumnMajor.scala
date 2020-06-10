package is.hail.types.encoded
import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, typeToTypeInfo}
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.{PCanonicalNDArray, PCanonicalNDArraySettable, PNDArrayValue, PType}
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
    val pnd = pt.asInstanceOf[PCanonicalNDArray]
    assert(pnd.elementType.required)
    val ndarray = coerce[Long](v)
    assert(ndarray.isInstanceOf[Settable[Long]])

    val writeShapes = (0 until nDims).map(i => out.writeLong(pnd.loadShape(ndarray, i)))
    // Note, encoded strides is in terms of indices into ndarray, not bytes.
    val writeStrides = (0 until nDims).map(i => out.writeLong(pnd.loadStride(ndarray, i) / pnd.elementType.byteSize))

    val writeElemF = elementType.buildEncoder(pnd.elementType, mb.ecb)

    val idxVars = (0 until nDims).map(i => mb.newLocal[Long]())
    val storeElement = mb.newLocal[Double]("nda_elem_out")

    val loadAndWrite = Code(
      storeElement := pnd.loadElementToIRIntermediate(idxVars, ndarray, mb).asInstanceOf[Code[Double]],
      Code._println(const("Indices ").concat(idxVars.map(_.toS.concat(" ")).reduce(_ concat _))),
      Code._println(storeElement.get.toS),
      writeElemF(storeElement, out)
    )


    val columnMajorLoops = idxVars.zipWithIndex.foldLeft(loadAndWrite) { case (innerLoops, (dimVar, dimIdx)) =>
      Code(
        dimVar := 0L,
        Code.whileLoop(dimVar < pnd.loadShape(ndarray, dimIdx),
          innerLoops,
          dimVar := dimVar + 1L
        )
      )
    }

    EmitCodeBuilder.scopedVoid(mb){cb =>
      cb.append(Code(writeShapes))
      // Print data array
      cb.append(out.writeInt(6))
      cb.append(columnMajorLoops)
    }
  }

  override def _buildDecoder(pt: PType, mb: EmitMethodBuilder[_], region: Value[Region], in: Value[InputBuffer]): Code[_] = {
    val pnd = pt.asInstanceOf[PCanonicalNDArray]
    val shapeVars = (0 until nDims).map(i => mb.newLocal[Long](s"shape_$i"))

    val arrayDecoder = EArray(elementType, true).buildDecoder(pnd.data.pType, mb.ecb)
    val dataAddress = mb.newLocal[Long]("data_addr")

    Code(
      Code(shapeVars.map(shapeVar => shapeVar := in.readLong())),
      dataAddress := arrayDecoder(region, in),
      pnd.construct(pnd.makeShapeBuilder(shapeVars), pnd.makeColumnMajorStridesBuilder(shapeVars, mb), dataAddress, mb)
    )
  }

  override def _buildSkip(mb: EmitMethodBuilder[_], r: Value[Region], in: Value[InputBuffer]): Code[Unit] = {
    val arraySkipper = EArray(elementType, true).buildSkip(mb)

    Code(
      Code((0 until nDims * 2).map(_ => in.skipLong())),
      arraySkipper(r, in)
    )
  }

  def _decodedPType(requestedType: Type): PType = {
    val requestedTNDArray = requestedType.asInstanceOf[TNDArray]
    val elementPType = elementType.decodedPType(requestedTNDArray.elementType)
    PCanonicalNDArray(elementPType, requestedTNDArray.nDims, required)
  }
  override def setRequired(required: Boolean): EType = ENDArrayColumnMajor(elementType, nDims, required)

  override def _asIdent = s"ndarray_of_${elementType.asIdent}"
  override def _toPretty = s"ENDArray[$elementType,$nDims]"
}
