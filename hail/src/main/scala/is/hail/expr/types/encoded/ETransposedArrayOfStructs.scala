package is.hail.expr.types.encoded

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.types.{BaseStruct, BaseType}
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
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

  override def _decodeCompatible(pt: PType): Boolean = pt match {
    case t: PArray =>
      if (!t.elementType.isInstanceOf[PBaseStruct])
        false
      else {
        val ps = t.elementType.asInstanceOf[PBaseStruct]
        ps.required == structRequired &&
          size >= ps.size &&
          ps.fields.forall { f =>
            hasField(f.name) && fieldType(f.name).decodeCompatible(f.typ)
          }
      }
    case _ => false
  }

  override def _encodeCompatible(pt: PType): Boolean = pt match {
    case t: PArray =>
      if (!t.elementType.isInstanceOf[PBaseStruct])
        false
      else {
        val ps = t.elementType.asInstanceOf[PBaseStruct]
        ps.required == structRequired &&
        size <= ps.size &&
        fields.forall { f =>
          ps.hasField(f.name) && f.typ.encodeCompatible(ps.fieldType(f.name))
        }
      }
    case _ => false
  }

  def _buildDecoder(pt: PType, mb: MethodBuilder, region: Value[Region], in: Value[InputBuffer]): Code[_] = {
    val pa = pt.asInstanceOf[PArray]
    val ps = pa.elementType.asInstanceOf[PBaseStruct]

    val len = mb.newField[Int]("len")
    val alen = mb.newField[Int]("alen")
    val i = mb.newLocal[Int]("i")
    val nMissing = mb.newLocal[Int]("nMissing")
    val anMissing = mb.newField[Int]("anMissing")
    val fmbytes = mb.newField[Long]("fmbytes")
    val array = mb.newLocal[Long]("array")

    val prefix = Code(
      len := in.readInt(),
      nMissing := pa.nMissingBytes(len),
      array := pa.allocate(region, len),
      pa.storeLength(array, len),
      if (structRequired)
        Code(
          alen := len,
          anMissing := nMissing
        )
      else
        Code(
          in.readBytes(region, array + const(pa.lengthHeaderBytes), nMissing),
          i := 0,
          alen := 0,
          Code.whileLoop(i < len,
            Code(
              alen := alen + pa.isElementDefined(array, i).toI,
              i := i + const(1))),
          anMissing := UnsafeUtils.packBitsToBytes(alen)),
      if (fields.forall(_.typ.required)) {
        fmbytes := 0
      } else {
        fmbytes := region.allocate(const(1), anMissing.toL)
      }
    )

    val decodeFields = Code(fields.grouped(64).zipWithIndex.map { case (fieldGroup, groupIdx) =>
      val groupMB = mb.fb.newMethod(s"read_fields_group_$groupIdx", Array[TypeInfo[_]](LongInfo, classInfo[Region], classInfo[InputBuffer]), UnitInfo)
      val arrayGrp = groupMB.getArg[Long](1)
      val regionGrp = groupMB.getArg[Region](2)
      val inGrp = groupMB.getArg[InputBuffer](3)
      val i = groupMB.newLocal[Int]("i")
      val j = groupMB.newLocal[Int]("j")

      val decoders = fieldGroup.map { ef =>
        ps.selfField(ef.name) match {
          case Some(pf) =>
            val inplaceDecode = ef.typ.buildInplaceDecoder(pf.typ, mb)
            val elem = pa.elementOffset(arrayGrp, len, i)
            val fld = ps.fieldOffset(elem, pf.index)
            if (ef.typ.required) {
              Code(
                i := 0,
                Code.whileLoop(i < len,
                  Code(
                    pa.isElementDefined(arrayGrp, i).mux(
                      inplaceDecode(regionGrp, fld, inGrp),
                      Code._empty),
                    i := i + const(1))))
            } else {
              Code(
                inGrp.readBytes(regionGrp, fmbytes, anMissing),
                i := 0,
                j := 0,
                Code.whileLoop(i < len,
                    Code(
                      pa.isElementDefined(arrayGrp, i).mux(
                        Code(
                          Region.loadBit(fmbytes, j.toL).mux(
                            ps.setFieldMissing(elem, pf.index),
                            Code(
                              ps.setFieldPresent(elem, pf.index),
                              inplaceDecode(regionGrp, fld, inGrp))),
                          j := j + const(1)),
                        Code._empty),
                      i := i + const(1))))
            }
          case None =>
            val skip = ef.typ.buildSkip(mb)
            if (ef.typ.required) {
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
      groupMB.invoke(array, region, in)
    }.toArray)

    Code(
      prefix,
      decodeFields,
      array
    )
  }

  def _buildSkip(mb: MethodBuilder, r: Value[Region], in: Value[InputBuffer]): Code[Unit] = {
    val len = mb.newLocal[Int]("len")
    val i = mb.newLocal[Int]("i")
    val nMissing = mb.newLocal[Int]("nMissing")
    val alen = mb.newLocal[Int]("alen")
    val anMissing = mb.newLocal[Int]("anMissing")
    val fmbytes = mb.newLocal[Long]("fmbytes")
    val skipFields = Code(fields.map { f =>
      val skip = f.typ.buildSkip(mb)
      if (f.typ.required) {
        Code(
          i := 0,
          Code.whileLoop(i < alen, Code(skip(r, in), i := i + 1)))
      } else {
        Code(
          in.readBytes(r, fmbytes, anMissing),
          i := 0,
          Code.whileLoop(i < alen,
            Code(
              Region.loadBit(fmbytes, i.toL).mux(
                Code._empty,
                skip(r, in)),
              i := i + 1)))
      }
    }.toArray)

    val prefix = Code(len := in.readInt(), nMissing := UnsafeUtils.packBitsToBytes(len))
    if (structRequired) {
      Code(
        prefix,
        alen := len,
        anMissing := UnsafeUtils.packBitsToBytes(alen),
        fmbytes := r.allocate(const(1), nMissing.toL),
        skipFields
      )
    } else {
      val mbytes = mb.newLocal[Long]("mbytes")
      Code(
        prefix,
        mbytes := r.allocate(const(1), nMissing.toL),
        in.readBytes(r, mbytes, nMissing),
        i := 0,
        alen := 0,
        Code.whileLoop(i < len,
          Code(
            alen := alen + (!Region.loadBit(mbytes, i.toL)).toI,
            i := i + const(1))),
        anMissing := UnsafeUtils.packBitsToBytes(alen),
        fmbytes := r.allocate(const(1), nMissing.toL),
        skipFields
      )
    }
  }

  def _buildEncoder(pt: PType, mb: MethodBuilder, v: Value[_], out: Value[OutputBuffer]): Code[Unit] = {
    val pa = pt.asInstanceOf[PArray]
    val ps = pa.elementType.asInstanceOf[PBaseStruct]

    val array = coerce[Long](v)
    val len = mb.newField[Int]("arrayLength")
    val i = mb.newLocal[Int]("i")
    val writeLen = out.writeInt(len)

    val writeMissingBytes =
      if (!structRequired) {
        out.writeBytes(array + const(pa.lengthHeaderBytes), pa.nMissingBytes(len))
      } else
        Code._empty

    val writeFields = Code(fields.grouped(64).zipWithIndex.map { case (fieldGroup, groupIdx) =>
      val groupMB = mb.fb.newMethod(s"write_fields_group_$groupIdx", Array[TypeInfo[_]](LongInfo, classInfo[OutputBuffer]), UnitInfo)
      val addr = groupMB.getArg[Long](1)
      val out2 = groupMB.getArg[OutputBuffer](2)

      val b = groupMB.newLocal[Int]("b")
      val j = groupMB.newLocal[Int]("j")
      val presentIdx = groupMB.newLocal[Int]("presentIdx")

      val encoders = fieldGroup.map { ef =>
        val fidx = ps.fieldIdx(ef.name)
        val pf = ps.fields(fidx)
        val encodeField = ef.typ.buildEncoder(pf.typ, groupMB)
        val elem = pa.elementOffset(addr, len, j)
        val f = ps.fieldOffset(elem, fidx)

        val writeMissingBytes = if (ef.typ.required)
          Code._empty
        else {
          Code(
          b := 0,
          j := 0,
          presentIdx := 0,
          Code.whileLoop(j < len,
            Code(
              pa.isElementDefined(addr, j).mux(
                Code(
                  b := b | (ps.isFieldMissing(elem, fidx).toI << (presentIdx & 7)),
                  presentIdx := presentIdx + const(1),
                  (presentIdx & 7).ceq(0).mux(
                    Code(out2.writeByte(b.toB), b := 0),
                    Code._empty)),
                Code._empty),
              j := j + const(1))),
          (presentIdx & 7).cne(0).mux(
              out2.writeByte(b.toB),
              Code._empty))
        }

        Code(
          writeMissingBytes,
          j := 0,
          Code.whileLoop(j < len,
            Code(
            pa.isElementDefined(addr, j).mux(
              ps.isFieldDefined(elem, fidx).mux(
                encodeField(Region.loadIRIntermediate(pf.typ)(f), out2),
                Code._empty),
              Code._empty),
              j := j + 1)))
      }.toArray

      groupMB.emit(Code(encoders))
      groupMB.invoke(addr, out)
    }.toArray)

    Code(
      i := 0,
      len := pa.loadLength(array),

      writeLen,
      writeMissingBytes,
      writeFields)
  }

  def _decodedPType(requestedType: Type): PType = requestedType match {
    case t: TDict =>
      val keyType = fieldType("key")
      val valueType = fieldType("value")
      PDict(keyType.decodedPType(t.keyType), valueType.decodedPType(t.valueType), required)
    case t: TIterable =>
      val pElementType = t.elementType match {
        case elem: TStruct => PStruct(elem.fields.map { case Field(name, typ, idx) =>
          PField(name, fieldType(name).decodedPType(typ), idx)
        }, structRequired)
        case elem: TTuple => PTuple(elem.fields.map { case Field(name, typ, idx) =>
          PTupleField(idx, fieldType(name).decodedPType(typ))
        }, structRequired)
        case elem: TLocus => PLocus(elem.rgBc, structRequired)
        case elem: TInterval =>
          val pointType = fieldType("start")
          require(pointType == fieldType("end"))
          PInterval(pointType.decodedPType(elem.pointType), structRequired)
        case elem: TNDArray =>
          val elementType = fieldType("data").asInstanceOf[EContainer].elementType
          PNDArray(elementType.decodedPType(elem.elementType), elem.nDims, structRequired)
      }
      t match {
        case _: TSet => PSet(pElementType, required)
        case _: TArray => PArray(pElementType, required)
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
}
