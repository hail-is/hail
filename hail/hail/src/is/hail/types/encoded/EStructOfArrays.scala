package is.hail.types.encoded

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, IEmitCode}
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces._
import is.hail.types.virtual.{Field => TField, _}
import is.hail.utils._

object EStructOfArrays {
  // expand this as more types are supported
  def supportsFieldType(typ: EType): Boolean = typ match {
    case _: EBoolean => true
    case _: EFloat32 => true
    case _: EFloat64 => true
    case _: EInt32 => true
    case _: EInt64 => true
    case _ => false
  }

  def supportsFieldType(typ: Type): Boolean = typ match {
    case TBoolean | TFloat32 | TFloat64 | TInt32 | TInt64 => true
    case _ => false
  }

  def fromTypeAndRequiredness(t: TIterable, r: RIterable): EStructOfArrays = {
    val et = tcoerce[TBaseStruct](t.elementType)
    val ret = tcoerce[RBaseStruct](r.elementType)
    val fields = et.fields.zip(ret.fields).map { case (TField(name, typ, index), r) =>
      val encodedType = typ match {
        case TBoolean => EArray(EBoolean(r.typ.required), required = true)
        case TFloat32 => EArray(EFloat32(r.typ.required), required = true)
        case TFloat64 => EArray(EFloat64(r.typ.required), required = true)
        case TInt32 => EArray(EInt32(r.typ.required), required = true)
        case TInt64 => EArray(EInt64(r.typ.required), required = true)
      }
      EField(name, encodedType, index)
    }

    EStructOfArrays(fields, required = r.required, structRequired = ret.required)
  }
}

final case class EStructOfArrays(
  fields: IndexedSeq[EField],
  val required: Boolean = false,
  val structRequired: Boolean = false,
) extends EContainer {
  assert(fields.zipWithIndex.forall { case (f, i) => f.index == i })
  require(fields.forall(f => f.typ.isInstanceOf[EContainer]))

  // We don't care about the requiredness of these EContainers, but at the top level,
  // they should all be present, so we require them to be required
  require(fields.forall(f => f.typ.required))

  val elementType = EBaseStruct(
    fields.map { field =>
      EField(field.name, field.typ.asInstanceOf[EContainer].elementType, field.index)
    },
    required = structRequired,
  )

  elementType.fields.foreach { fld =>
    if (!EStructOfArrays.supportsFieldType(fld.typ))
      throw new UnsupportedOperationException(
        s"${fld.typ} not supported as field type for struct of arrays"
      )
  }

  override def _decodedSType(requestedType: Type): SType = {
    require(requestedType.isInstanceOf[TArray])
    val elementPType = elementType.decodedPType(requestedType.asInstanceOf[TContainer].elementType)
    SIndexablePointer(PCanonicalArray(elementPType, required = false))
  }

  override def _buildDecoder(
    cb: EmitCodeBuilder,
    t: Type,
    region: Value[Region],
    in: Value[InputBuffer],
  ): SValue = {
    val st = tcoerce[SIndexablePointer](decodedSType(t))
    val pt = tcoerce[PCanonicalArray](st.pType)
    val ept = tcoerce[PBaseStruct](pt.elementType)
    assert(
      pt.elementType.required == elementType.required,
      s"${pt.elementType.required} | ${elementType.required}",
    )

    val scratchRegion: Value[Region] = cb.memoize(region.getPool().invoke[Region]("getRegion"))
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
      cb += in.readBytes(region, arrayPtr + pt.missingBytesOffset, nMissingBytes)

    // scratch variables
    val i = cb.newLocal[Int]("i")
    val structPtr = cb.newLocal[Long]("element_out") // pointer to struct we're constructing
    fields.foreach { field =>
      cb += scratchRegion.clearRegion()
      if (!ept.hasField(field.name)) {
        field.typ.buildSkip(cb.emb.ecb)(cb, scratchRegion, in)
      } else {
        val pFieldIdx = ept.fieldIdx(field.name)
        val pFieldType = tcoerce[PPrimitive](ept.types(pFieldIdx))

        val array = field.typ.buildDecoder(t, cb.emb.ecb)(cb, scratchRegion, in).asIndexable
        cb.if_(
          array.loadLength.cne(length),
          cb._fatal(
            "Mismatch in length for decoded array of field `",
            field.name,
            "` expected ",
            length.toS,
            ", was ",
            array.loadLength.toS,
          ),
        )
        cb.assign(i, 0)
        cb.assign(structPtr, pt.firstElementOffset(arrayPtr))
        cb.while_(
          i < length, {
            cb.if_(
              pt.isElementMissing(arrayPtr, i),
              if (!pFieldType.required) ept.setFieldMissing(cb, structPtr, pFieldIdx),
              array.loadElement(cb, i).consume(
                cb,
                if (!pFieldType.required) ept.setFieldMissing(cb, structPtr, pFieldIdx),
                { sv =>
                  val fieldPtr = ept.fieldOffset(structPtr, pFieldIdx)
                  pFieldType.storeAtAddress(cb, fieldPtr, region, sv, true)
                },
              ),
            )
            cb.assign(i, i + 1)
            cb.assign(structPtr, pt.nextElementAddress(structPtr))
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

  override def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit =
    v match {
      case sv: SIndexablePointerValue =>
        val pArray = sv.st.pType.asInstanceOf[PCanonicalArrayBackedContainer].arrayRep
        val r: Value[Region] = // scratch region
          cb.memoize(cb.emb.ecb.pool().invoke[Region]("getRegion"))
        cb += out.writeInt(sv.length)
        if (!elementType.required) {
          val nMissingBytes = cb.memoize(pArray.nMissingBytes(sv.length))
          cb += out.writeBytes(sv.a + pArray.missingBytesOffset, nMissingBytes)
        }

        fields.foreach { field =>
          val pFieldType = tcoerce[PBaseStruct](pArray.elementType).fieldByName(field.name).typ
          require(EStructOfArrays.supportsFieldType(pFieldType.virtualType))
          val arrayType = PCanonicalArray(pFieldType, required = false)
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
    val arrayValue =
      arrayType.constructFromElements(cb, r, sv.length, deepCopy = false) { (cb, i) =>
        if (arrayType.elementType.required) {
          sv.loadElement(cb, i).consumeI(
            cb, {
              val elementSType = tcoerce[SBaseStruct](sv.st.elementType)
              val fieldSType = elementSType.fieldTypes(elementSType.fieldIdx(field.name))
              IEmitCode.present(
                cb,
                cb.newSLocal(
                  fieldSType,
                  s"${fieldSType.asIdent}_present_for_required_element_in_missing_struct",
                ),
              )
            },
            { case sbsv: SBaseStructValue =>
              sbsv.loadField(cb, field.name)
            },
          )
        } else {
          sv.loadElement(cb, i).flatMap(cb) { case sbsv: SBaseStructValue =>
            sbsv.loadField(cb, field.name)
          }
        }
      }

    val arrayEncoder = field.typ.buildEncoder(arrayValue.st, cb.emb.ecb)
    arrayEncoder(cb, arrayValue, out)
  }

  override def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit = {
    val length = cb.memoize(in.readInt())
    val nMissingBytes =
      cb.memoize(UnsafeUtils.packBitsToBytes(length)) // valid for all top level arrays
    if (!elementType.required)
      cb += in.skipBytes(nMissingBytes)

    fields.foreach(field => field.typ.buildSkip(cb.emb.ecb)(cb, r, in))
  }

  override def setRequired(newRequired: Boolean): EStructOfArrays =
    EStructOfArrays(fields, required = newRequired, structRequired = structRequired)

  override def _asIdent: String = s"struct_of_arrays_from_${elementType.asIdent}"

  override def _toPretty: String = {
    val sb = new StringBuilder
    _pretty(sb, 0, compact = true)
    sb.result()
  }

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = {
    sb ++= "EStructOfArrays["
    if (structRequired) sb += '+'
    sb += '{'
    if (compact) {
      fields.foreachBetween(_.pretty(sb, indent, compact))(sb += ',')
    } else if (fields.isEmpty) {
      sb += ' '
    } else {
      sb += '\n'
      fields.foreachBetween(_.pretty(sb, indent + 4, compact))(sb ++= ",\n")
      sb += '\n'
      sb ++= (" " * indent)
    }
    sb ++= "}]"
  }
}
