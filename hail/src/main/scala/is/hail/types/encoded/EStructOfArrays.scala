package is.hail.types.encoded

import is.hail.annotations.{UnsafeUtils, Region}
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.tcoerce
import is.hail.types.physical._
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.virtual._
import is.hail.utils._

object EStructOfArrays {
  // expand this as more types are supported
  def supportsFieldType(typ: EType): Unit = typ match {
    case _: EInt32 =>
    case _ => throw new UnsupportedOperationException(
        s"$typ not supported as field type for struct of arrays"
      )
  }

  def elementSize(fieldType: EType): Long = fieldType match {
    case _: EInt32 | _: EFloat32 => 4
    case _: EInt64 | _: EFloat64 => 8
    case _: EBoolean => 1
  }
}

final case class EStructOfArrays(
  override val elementType: EBaseStruct,
  val required: Boolean = false,
) extends EContainer {
  elementType.fields.foreach(fld => EStructOfArrays.supportsFieldType(fld.typ))

  // length: 1 int32
  // struct mbits: (length + 7) >>> 3 bytes [if optional]
  // for each field [only int32 supported for now]:
  //    // top level length elided
  //    field mbits: (length + 7) >>> 3 bytes [if optional]
  //    elements: length int32
  //
  // unlike EArray, missing elements are zeroed in the stream, rather than
  // being not written, this should enable faster skips and possibly larger
  // chunked reads
  //
  // it is a bug for a field missing bit to be unset (element present)
  // if the struct missing bit is set (element missing), that is,
  // struct_mbits & field_mbits == field_mbits

  def _buildDecoder(cb: EmitCodeBuilder, t: Type, region: Value[Region], in: Value[InputBuffer])
    : SValue = {
    val st: SIndexablePointer = tcoerce(decodedSType(t))
    val pt: PCanonicalArray = tcoerce(st.pType)
    val ept: PBaseStruct = tcoerce(pt.elementType)
    assert(
      pt.elementType.required == elementType.required,
      s"${pt.elementType.required} | ${elementType.required}",
    )

    val scratchRegion: Value[Region] = cb.memoize(cb.emb.ecb.pool().invoke[Region]("getRegion"))
    val length = cb.memoize(in.readInt())

    val arrayPtr = cb.memoize(pt.allocate(region, length))
    cb += Region.setMemory(
      arrayPtr,
      pt.pastLastElementOffset(arrayPtr, length) - arrayPtr,
      0.toByte,
    )
    pt.storeLength(cb, arrayPtr, length)
    val nMissingBytes = cb.memoize(pt.nMissingBytes(length))
    if (!elementType.required)
      cb += in.readBytes(region, arrayPtr + pt.missingBytesOffset, pt.nMissingBytes(length))

    // scratch variables
    val i = cb.newLocal[Int]("i")
    val structPtr = cb.newLocal[Long]("element_out") // pointer to struct we're constructing
    val elementPtr = cb.newLocal[Long]("element_in") // pointer to element we're reading
    elementType.fields.foreach { field =>
      val fieldIdx = ept.fieldIdx(field.name)
      cb += scratchRegion.clearRegion()
      if (!ept.hasField(field.name)) {
        skipField(cb, field, length, nMissingBytes, in)
      } else {
        // TODO, pseudocode
        // in <- passed in input buffer
        // array <- already allocated output array
        // size <- elementSize(field.typ)
        // mbits <- allocate(n=nMissingBytes, a=1)  // if necessary
        // elements <- allocate(n=length * size, a=size)
        // in.read(to=mbits, amt=nMissingBytes)
        // in.read(to=elements, amt=length * size)
        // for i in range(length):
        //   if array.isElementMissing(i) || mbits.isMissing(i):
        //     array[i].setFieldMissing(field.name)
        //   else:
        //     array[i].setFieldPresent(field.name)
        //     array[i].setField(field.name, elements[i])
        val elementSize = const(EStructOfArrays.elementSize(field.typ))
        val arraySize = cb.memoize(elementSize * length.toL)
        val mbytes = scratchRegion.allocate(1L, nMissingBytes.toL)
        val elements = scratchRegion.allocate(elementSize, arraySize)
        cb += in.readBytes(scratchRegion, mbytes, nMissingBytes)
        cb += in.readBytes(scratchRegion, elements, arraySize.toI)
        cb.for_(
          {
            cb.assign(i, 0)
            cb.assign(structPtr, pt.firstElementOffset(arrayPtr))
            cb.assign(elementPtr, elements)
          },
          i < length, {
            cb.assign(i, i + 1)
            cb.assign(structPtr, pt.nextElementAddress(structPtr))
            cb.assign(elementPtr, elementPtr + elementSize)
          }, {
            cb.if_(
              pt.isElementMissing(arrayPtr, i) || Region.loadBit(mbytes, i.toL),
              ept.setFieldMissing(cb, structPtr, fieldIdx),
              // TODO set the struct value
              ept.setFieldPresent(cb, structPtr, fieldIdx),
            )
          },
        )
      }
    }

    cb += scratchRegion.invalidate()
    new SIndexablePointerValue(
      st,
      arrayPtr,
      length,
      cb.memoize(pt.firstElementOffset(arrayPtr, length)),
    )
  }

  def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit = v match {
    case sv: SIndexablePointerValue =>
      val pArray = sv.st.pType match {
        case t: PCanonicalArray => t
        case t: PCanonicalSet => t.arrayRep
        case t: PCanonicalDict => t.arrayRep
      }
      val r: Value[Region] = // scratch region
        cb.memoize(cb.emb.ecb.pool().invoke[Region]("getRegion"))
      cb += out.writeInt(sv.length)
      if (!elementType.required) {
        val nMissingBytes = cb.memoize(pArray.nMissingBytes(sv.length))
        cb += out.writeBytes(sv.a + pArray.missingBytesOffset, nMissingBytes)
      }

      elementType.fields.foreach { field =>
        val fieldPType = tcoerce[PBaseStruct](pArray.elementType).fieldByName(field.name).typ
        require(fieldPType.isPrimitive)
        val arrayType = PCanonicalArray(fieldPType, required = false)
        transposeAndWriteField(cb, field, arrayType, sv, r, out)
      }

      cb += r.invalidate()
  }

  private[this] def transposeAndWriteField(
    cb: EmitCodeBuilder,
    field: EField,
    arrayType: PCanonicalArray,
    sv: SIndexablePointerValue,
    r: Value[Region],
    out: Value[OutputBuffer],
  ): Unit = {
    cb += r.clearRegion()
    val length = sv.length
    val arrayPtr = cb.memoize(arrayType.zeroes(cb, r, length))
    sv.forEachDefinedOrMissing(cb)(
      (cb, idx) => PContainer.unsafeSetElementMissing(cb, arrayType, arrayPtr, idx),
      { case (cb, idx, sbsv: SBaseStructValue) =>
        sbsv.loadField(cb, field.name).consume(
          cb,
          PContainer.unsafeSetElementMissing(cb, arrayType, arrayPtr, idx),
          { fieldValue =>
            arrayType.elementType.storeAtAddress(
              cb,
              arrayType.elementOffset(arrayPtr, length, idx),
              r,
              fieldValue,
              deepCopy = false,
            )
          },
        )
      },
    )

    if (!field.typ.required) {
      cb += out.writeBytes(arrayPtr + arrayType.missingBytesOffset, arrayType.nMissingBytes(length))
    }

    val writeFrom = cb.memoize(arrayType.firstElementOffset(arrayPtr, length))
    cb += out.writeBytes(writeFrom, arrayType.contentsByteSize(length).toI)
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit = {
    val length = cb.memoize(in.readInt())
    val nMissingBytes =
      cb.memoize(UnsafeUtils.packBitsToBytes(length)) // valid for all top level arrays
    if (!elementType.required)
      cb += in.skipBytes(nMissingBytes)

    elementType.fields.foreach(field => skipField(cb, field, length, nMissingBytes, in))
  }

  private[this] def skipField(
    cb: EmitCodeBuilder,
    field: EField,
    length: Value[Int],
    nMissingBytes: Value[Int],
    in: Value[InputBuffer],
  ): Unit = {
    if (!field.typ.required)
      cb += in.skipBytes(nMissingBytes)
    in.skipBytes(length * EStructOfArrays.elementSize(field.typ).toInt)
  }

  def _decodedSType(requestedType: Type): SType = {
    require(requestedType.isInstanceOf[TArray])
    val elementPType = elementType.decodedPType(requestedType.asInstanceOf[TContainer].elementType)
    SIndexablePointer(PCanonicalArray(elementPType, required = false))
  }

  def setRequired(newRequired: Boolean): EStructOfArrays = EStructOfArrays(elementType, newRequired)

  def _asIdent: String = s"struct_of_arrays_from_${elementType.asIdent}"

  def _toPretty: String = {
    val sb = new StringBuilder
    _pretty(sb, 0, compact = true)
    sb.result()
  }

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = {
    sb.append("EStructOfArrays[")
    elementType.pretty(sb, indent, compact)
    sb += ']'
  }
}
