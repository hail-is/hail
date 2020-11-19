package is.hail.types.encoded

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, ParamType}
import is.hail.types.{BaseStruct, BaseType}
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

final case class ETransposedArrayOfStructs(
  fields: IndexedSeq[EField],
  override val required: Boolean = false, // is the array required
  structRequired: Boolean = false // are the structs required
) extends EType {

  def size: Int = fields.length
  val fieldIdx: Map[String, Int] = fields.map(f => (f.name, f.index)).toMap
  def field(name: String): EField = fields(fieldIdx(name))
  def fieldType(name: String): EType = field(name).typ
  def hasField(name: String): Boolean = fieldIdx.contains(name)

  def _decodeCompatible(pt: PType): Boolean = pt match {
    case t: PArray =>
      t.elementType match {
        case struct: PBaseStruct =>
          struct.required == structRequired &&
            size >= struct.size &&
            struct.fields.forall { f =>
              hasField(f.name) && fieldType(f.name).decodeCompatible(f.typ)
            }
        case _ => false
      }
    case _ => false
  }

  def _encodeCompatible(pt: PType): Boolean = pt match {
    case t: PArray =>
      t.elementType match {
        case struct: PBaseStruct =>
          struct.required == structRequired &&
            size <= struct.size &&
            fields.forall { f =>
              struct.hasField(f.name) && f.typ.encodeCompatible(struct.fieldType(f.name))
            }
        case _ => false
      }
    case _ => false
  }

  def _buildDecoder(cb: EmitCodeBuilder, pt: PType, region: Value[Region], in: Value[InputBuffer])(implicit line: LineNumber): Code[_] = {
    val mb = cb.emb
    val arrayPType = pt.asInstanceOf[PArray]
    val elementPStruct = arrayPType.elementType.asInstanceOf[PBaseStruct]

    val len = mb.cb.genFieldThisRef[Int]("len")
    val alen = mb.genFieldThisRef[Int]("alen")
    val i = mb.newLocal[Int]("i")
    val nMissing = mb.newLocal[Int]("nMissing")
    val anMissing = mb.cb.genFieldThisRef[Int]("anMissing")
    val fmbytes = mb.cb.genFieldThisRef[Long]("fmbytes")
    val array = mb.newLocal[Long]("array")

    val prefix = Code(
      len := in.readInt(),
      nMissing := arrayPType.nMissingBytes(len),
      array := arrayPType.allocate(region, len),
      arrayPType.storeLength(array, len),
      if (structRequired)
        Code(
          alen := len,
          anMissing := nMissing
        )
      else
        Code(
          in.readBytes(region, array + const(arrayPType.lengthHeaderBytes), nMissing),
          i := 0,
          alen := 0,
          Code.whileLoop(i < len,
            Code(
              alen := alen + arrayPType.isElementDefined(array, i).toI,
              i := i + const(1))),
          anMissing := UnsafeUtils.packBitsToBytes(alen)),
      if (fields.forall(_.typ.required)) {
        fmbytes := 0L
      } else {
        fmbytes := region.allocate(const(1L), anMissing.toL)
      }
    )

    val decodeFields = Code(fields.grouped(64).zipWithIndex.map { case (fieldGroup, groupIdx) =>
      val groupMB = mb.ecb.newEmitMethod(s"read_fields_group_$groupIdx", FastIndexedSeq[ParamType](LongInfo, classInfo[Region], classInfo[InputBuffer]), UnitInfo)
      val arrayGrp = groupMB.getCodeParam[Long](1)
      val regionGrp = groupMB.getCodeParam[Region](2)
      val inGrp = groupMB.getCodeParam[InputBuffer](3)
      val i = groupMB.newLocal[Int]("i")
      val j = groupMB.newLocal[Int]("j")

      val decoders = fieldGroup.map { encodedField =>
        elementPStruct.selfField(encodedField.name) match {
          case Some(pf) =>
            val inplaceDecode = encodedField.typ.buildInplaceDecoder(pf.typ, groupMB.ecb)
            val elem = () => arrayPType.elementOffset(arrayGrp, len, i)
            if (encodedField.typ.required) {
              Code(
                i := 0,
                Code.whileLoop(i < len,
                  Code(
                    arrayPType.isElementDefined(arrayGrp, i).orEmpty(
                      inplaceDecode(regionGrp, elementPStruct.fieldOffset(elem(), pf.index), inGrp)),
                    i := i + const(1))))
            } else {
              Code(
                inGrp.readBytes(regionGrp, fmbytes, anMissing),
                i := 0,
                j := 0,
                Code.whileLoop(i < len,
                    Code(
                      arrayPType.isElementDefined(arrayGrp, i).orEmpty(
                        Code(
                          Region.loadBit(fmbytes, j.toL).mux(
                            elementPStruct.setFieldMissing(elem(), pf.index),
                            Code(
                              elementPStruct.setFieldPresent(elem(), pf.index),
                              inplaceDecode(regionGrp, elementPStruct.fieldOffset(elem(), pf.index), inGrp))),
                          j := j + const(1))),
                      i := i + const(1))))
            }
          case None =>
            val skip = encodedField.typ.buildSkip(mb)
            if (encodedField.typ.required) {
              Code(
                i := 0,
                Code.whileLoop(i < alen, Code(skip(regionGrp, inGrp), i := i + const(1)))
              )
            } else {
              Code(
                inGrp.readBytes(regionGrp, fmbytes, anMissing),
                i := 0,
                Code.whileLoop(i < alen,
                  Code(
                    Region.loadBit(fmbytes, i.toL).mux(
                      Code._empty,
                      skip(regionGrp, inGrp)),
                    i := i + 1)))
            }
        }
      }.toArray

      groupMB.emit(Code(decoders))
      groupMB.invokeCode(array, region, in)
    }.toArray)

    Code(
      prefix,
      decodeFields,
      array
    )
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer])(implicit line: LineNumber): Unit = {
    val len = cb.newLocal[Int]("len")
    val i = cb.newLocal[Int]("i")
    val nMissing = cb.newLocal[Int]("nMissing")
    val alen = cb.newLocal[Int]("alen")
    val anMissing = cb.newLocal[Int]("anMissing")
    val mbytes = cb.newLocal[Long]("mbytes")

    cb.assign(len, in.readInt())
    cb.assign(nMissing, UnsafeUtils.packBitsToBytes(len))
    cb.assign(mbytes, r.allocate(const(1), nMissing.toL))
    val prefix = Code(len := in.readInt(), nMissing := UnsafeUtils.packBitsToBytes(len))
    if (structRequired) {
      cb.assign(alen, len)
    } else {
      cb += in.readBytes(r, mbytes, nMissing)
      cb.forLoop({
        cb.assign(i, 0)
        cb.assign(alen, 0)
      }, i < len,
        cb.assign(i, i + 1),
        cb.assign(alen, alen + (!Region.loadBit(mbytes, i.toL)).toI))
    }
    cb.assign(anMissing, UnsafeUtils.packBitsToBytes(alen))

    fields.foreach { f =>
      val skip = f.typ.buildSkip(cb.emb)
      if (f.typ.required) {
        cb.forLoop(cb.assign(i, 0), i < alen, cb.assign(i, i + 1), cb += skip(r, in))
      } else {
        cb += in.readBytes(r, mbytes, anMissing)
        cb.forLoop(cb.assign(i, 0), i < alen, cb.assign(i, i + 1),
          cb.ifx(!Region.loadBit(mbytes, i.toL), cb += skip(r, in)))
      }
    }
  }

  def _buildEncoder(cb: EmitCodeBuilder, pt: PType, v: Value[_], out: Value[OutputBuffer])(implicit line: LineNumber): Unit = {
    val arrayPType = pt.asInstanceOf[PArray]
    val elementPStruct = arrayPType.elementType.asInstanceOf[PBaseStruct]
    val array = PCode(pt, v).asIndexable.memoize(cb, "array")
    val len = array.loadLength()

    cb += out.writeInt(len)

    if (!structRequired) {
      cb += out.writeBytes(array.tcode[Long] + const(arrayPType.lengthHeaderBytes),
        arrayPType.nMissingBytes(len))
    }

    val b = cb.newLocal[Int]("b", 0)
    val j = cb.newLocal[Int]("j", 0)
    val presentIdx = cb.newLocal[Int]("presentIdx", 0)
    val pbss = cb.emb.newPLocal("struct", elementPStruct)
    def pbsv: PBaseStructValue = pbss.asInstanceOf[PBaseStructValue]
    fields.foreach { ef =>
      val fidx = elementPStruct.fieldIdx(ef.name)
      val pf = elementPStruct.fields(fidx)
      val encodeFieldF = ef.typ.buildEncoder(pf.typ, cb.emb.ecb)

      if (!ef.typ.required) {
        cb.forLoop(cb.assign(j, 0), j < len, cb.assign(j, j + 1), {
          array.loadElement(cb, j).consume(cb, { /* do nothing */ }, { pbsc =>
            // FIXME may be bad if structs get memoized into their fields, otherwise
            // probably fine
            cb.assign(pbss, pbsc)
            cb.assign(b, b | (pbsv.isFieldMissing(fidx).toI << (presentIdx & 7)))
            cb.assign(presentIdx, presentIdx + 1)
            cb.ifx((presentIdx & 7).ceq(0), {
              out.writeByte(b.toB)
              cb.assign(b, 0)
            })
          })
        })
      }

      cb.forLoop(cb.assign(j, 0), j < len, cb.assign(j, j + 1), {
        array.loadElement(cb, j).flatMap(cb) { pbsc =>
          cb.assign(pbss, pbsc)
          pbsv.loadField(cb, fidx)
        }
        .consume(cb, { /* do nothing */ }, { pc =>
          cb += encodeFieldF(pc.code, out)
        })
      })
    }
  }

  def _decodedPType(requestedType: Type): PType = requestedType match {
    case t: TDict =>
      val keyType = fieldType("key")
      val valueType = fieldType("value")
      PCanonicalDict(keyType.decodedPType(t.keyType), valueType.decodedPType(t.valueType), required)
    case t: TIterable =>
      val pElementType = t.elementType match {
        case elem: TStruct => PCanonicalStruct(elem.fields.map { case Field(name, typ, idx) =>
          PField(name, fieldType(name).decodedPType(typ), idx)
        }, structRequired)
        case elem: TTuple => PCanonicalTuple(elem.fields.map { case Field(name, typ, idx) =>
          PTupleField(idx, fieldType(name).decodedPType(typ))
        }, structRequired)
        case elem: TLocus => PCanonicalLocus(elem.rgBc, structRequired)
        case elem: TInterval =>
          val pointType = fieldType("start")
          require(pointType == fieldType("end"))
          PCanonicalInterval(pointType.decodedPType(elem.pointType), structRequired)
        case elem: TNDArray =>
          val elementType = fieldType("data").asInstanceOf[EContainer].elementType
          PCanonicalNDArray(elementType.decodedPType(elem.elementType), elem.nDims, structRequired)
      }
      t match {
        case _: TSet => PCanonicalSet(pElementType, required)
        case _: TArray => PCanonicalArray(pElementType, required)
      }
  }

  def _asIdent: String = {
    val sb = new StringBuilder
    sb.append(s"transposed_array_of_${if (structRequired) "r" else "o"}_struct_of_")
    fields.foreachBetween { f =>
      sb.append(f.typ.asIdent)
    } {
      sb.append("AND")
    }
    sb.append("END")
    sb.result()
  }

  def _toPretty: String = {
    val sb = new StringBuilder
    _pretty(sb, 0, compact = true)
    sb.result()
  }

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean) {
    if (compact) {
      sb.append(s"ETransposedArrayOfStructs[${if (structRequired) "True" else "False"}]{")
      fields.foreachBetween(_.pretty(sb, indent, compact))(sb += ',')
      sb += '}'
    } else {
      if (fields.length == 0)
        sb.append("ETransposedArrayOfStructs { }")
      else {
        sb.append("ETransposedArrayOfStructs {")
        sb += '\n'
        fields.foreachBetween(_.pretty(sb, indent + 4, compact))(sb.append(",\n"))
        sb += '\n'
        sb.append(" " * indent)
        sb += '}'
      }
    }
  }

  def setRequired(newRequired: Boolean): ETransposedArrayOfStructs =
    ETransposedArrayOfStructs(fields, newRequired, structRequired)
}
