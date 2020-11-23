package is.hail.types.encoded

import is.hail.annotations.{Region}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder}
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.{PCanonicalNDArray, PCode, PType}
import is.hail.types.virtual.{TNDArray, Type}
import is.hail.utils._

case class ENDArrayColumnMajor(elementType: EType, nDims: Int, required: Boolean = false) extends EContainer {

  override def decodeCompatible(pt: PType): Boolean = {
    pt.required == required &&
      pt.isInstanceOf[PCanonicalNDArray] &&
      elementType.decodeCompatible(pt.asInstanceOf[PCanonicalNDArray].elementType)
  }

  override def encodeCompatible(pt: PType): Boolean = {
    pt.required == required &&
      pt.isInstanceOf[PCanonicalNDArray] &&
      elementType.encodeCompatible(pt.asInstanceOf[PCanonicalNDArray].elementType)
  }

  def _buildEncoder(cb: EmitCodeBuilder, pt: PType, v: Value[_], out: Value[OutputBuffer]): Unit = {
    val ndarray = PCode(pt, v).asNDArray.memoize(cb, "ndarray")
    val writeElemF = elementType.buildEncoder(ndarray.pt.elementType, cb.emb.ecb)

    val shapes = ndarray.shapes(cb)
    shapes.foreach(s => cb += out.writeLong(s))

    val idxVars = Array.tabulate(ndarray.pt.nDims)(i => cb.newLocal[Long](s"idx_$i"))
    cb += idxVars.zipWithIndex.foldLeft(writeElemF(ndarray.loadElement(idxVars, cb).asPCode.code, out))
    { case (innerLoops, (dimVar, dimIdx)) =>
      Code(
        dimVar := 0L,
        Code.whileLoop(dimVar < shapes(dimIdx),
          innerLoops,
          dimVar := dimVar + 1L
        )
      )
    }
  }

  def _buildDecoder(
    cb: EmitCodeBuilder,
    pt: PType,
    region: Value[Region],
    in: Value[InputBuffer]
  ): Code[_] = {
    val pnd = pt.asInstanceOf[PCanonicalNDArray]
    val readElemF = elementType.buildInplaceDecoder(pnd.elementType, cb.emb.ecb)

    val shapeVars = (0 until nDims).map(i => cb.newLocal[Long](s"ndarray_decoder_shape_$i", in.readLong()))
    val totalNumElements = cb.newLocal[Long]("ndarray_decoder_total_num_elements", 1L)
    shapeVars.foreach { s =>
      cb.assign(totalNumElements, totalNumElements * s)
    }

    val dataAddress = cb.newLocal[Long]("ndarray_decoder_data_addr",
      pnd.data.pType.allocate(region, totalNumElements.toI))
    cb += pnd.data.pType.stagedInitialize(dataAddress, totalNumElements.toI)

    val dataIdx = cb.newLocal[Int]("ndarray_decoder_data_idx")
    cb.forLoop(cb.assign(dataIdx, 0), dataIdx < totalNumElements.toI, cb.assign(dataIdx, dataIdx + 1),
      cb += readElemF(region, pnd.data.pType.elementOffset(dataAddress, totalNumElements.toI, dataIdx), in))

    pnd.construct(pnd.makeShapeBuilder(shapeVars),
      pnd.makeColumnMajorStridesBuilder(shapeVars, cb.emb),
      dataAddress, cb.emb, region)
      .tcode[Long]
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit = {
    val skip = elementType.buildSkip(cb.emb)

    val numElements = cb.newLocal[Long]("ndarray_skipper_total_num_elements",
      (0 until nDims).foldLeft(const(1L).get){ (p, i) => p * in.readLong() })
    val i = cb.newLocal[Long]("ndarray_skipper_data_idx")
    cb.forLoop(cb.assign(i, 0L), i < numElements, cb.assign(i, i + 1L), cb += skip(r, in))
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
