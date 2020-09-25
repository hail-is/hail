package is.hail.types.encoded

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitCodeBuilder, EmitMethodBuilder, ParamType}
import is.hail.types.{BaseStruct, BaseType}
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

final case class EField(name: String, typ: EType, index: Int) {
  def pretty(sb: StringBuilder, indent: Int, compact: Boolean) {
    if (compact) {
      sb.append(prettyIdentifier(name))
      sb.append(":")
    } else {
      sb.append(" " * indent)
      sb.append(prettyIdentifier(name))
      sb.append(": ")
    }
    typ.pretty(sb, indent, compact)
  }
}

final case class EBaseStruct(fields: IndexedSeq[EField], override val required: Boolean = false) extends EFundamentalType {
  assert(fields.zipWithIndex.forall { case (f, i) => f.index == i })

  val types: Array[EType] = fields.map(_.typ).toArray
  def size: Int = types.length
  val fieldNames: Array[String] = fields.map(_.name).toArray
  val fieldIdx: Map[String, Int] = fields.map(f => (f.name, f.index)).toMap
  def hasField(name: String): Boolean = fieldIdx.contains(name)
  def fieldType(name: String): EType = types(fieldIdx(name))
  val (missingIdx: Array[Int], nMissing: Int) = BaseStruct.getMissingIndexAndCount(types.map(_.required))
  val nMissingBytes = UnsafeUtils.packBitsToBytes(nMissing)

  if (!fieldNames.areDistinct()) {
    val duplicates = fieldNames.duplicates()
    fatal(s"cannot create struct with duplicate ${plural(duplicates.size, "field")}: " +
      s"${fieldNames.map(prettyIdentifier).mkString(", ")}", fieldNames.duplicates())
  }

  override def _decodeCompatible(pt: PType): Boolean = {
    val pt2 = if (pt.isInstanceOf[PNDArray]) pt.asInstanceOf[PNDArray].representation else pt

    if (!pt2.isInstanceOf[PBaseStruct])
      false
    else {
      val ps = pt2.asInstanceOf[PBaseStruct]
      ps.required <= required &&
        ps.size <= size &&
        ps.fields.forall { f =>
          hasField(f.name) && fieldType(f.name).decodeCompatible(f.typ)
        }
    }
  }

  override def _encodeCompatible(pt: PType): Boolean = {
    if (!pt.isInstanceOf[PBaseStruct])
      false
    else {
      val ps = pt.asInstanceOf[PBaseStruct]
      ps.required >= required &&
        size <= ps.size &&
        fields.forall { f =>
          ps.hasField(f.name) && f.typ.encodeCompatible(ps.fieldType(f.name))
        }
    }
  }

  def _decodedPType(requestedType: Type): PType = requestedType match {
    case t: TInterval =>
      val repr = t.representation
      val pointType = _decodedPType(repr).asInstanceOf[PStruct].fieldType("start")
      PCanonicalInterval(pointType, required)
    case t: TLocus => PCanonicalLocus(t.rg, required)
    case t: TStruct =>
      val pFields = t.fields.map { case Field(name, typ, idx) =>
        val pt = fieldType(name).decodedPType(typ)
        PField(name, pt, idx)
      }
      PCanonicalStruct(pFields, required)
    case t: TTuple =>
      val pFields = t.fields.map { case Field(name, typ, idx) =>
        val pt = fieldType(name).decodedPType(typ)
        PTupleField(idx, pt)
      }
      PCanonicalTuple(pFields, required)
    case t: TNDArray =>
      val elementType = _decodedPType(t.representation).asInstanceOf[PStruct].fieldType("data").asInstanceOf[PArray].elementType
      PCanonicalNDArray(elementType, t.nDims, required)
  }

  def _buildFundamentalEncoder(cb: EmitCodeBuilder, pt: PType, v: Value[_], out: Value[OutputBuffer]): Unit = {
    val pv = PCode(pt, v).asBaseStruct.memoize(cb, "base_struct")
    val ft = pv.pt
    // write missing bytes
    if (ft.size == size) {
      val missingBytes = UnsafeUtils.packBitsToBytes(ft.nMissing)
      ft match {
        case ps: PCanonicalBaseStruct if ps.fieldRequired.sameElements(fields.map(_.typ.required)) =>
          if (nMissingBytes > 1)
            cb += out.writeBytes(pv.tcode[Long], missingBytes - 1)
          if (nMissingBytes > 0)
            cb += out.writeByte((Region.loadByte(pv.tcode[Long] + (missingBytes.toLong - 1)).toI & const(EType.lowBitMask(ft.nMissing & 0x7))).toB)
        case _ =>
          fields.filter(f => !f.typ.required)
            .grouped(8)
            .foreach { group =>
              val byte = group.zipWithIndex.map { case (f, i) =>
                pv.isFieldMissing(f.index).toI << i
              }.reduce(_ | _).toB
              cb += out.writeByte(byte)
            }
      }
    } else {
      var j = 0
      var n = 0
      while (j < size) {
        var b: Code[Int] = 0
        var k = 0
        while (k < 8 && j < size) {
          val f = fields(j)
          if (!f.typ.required) {
            b = b | (pv.isFieldMissing(f.name).toI << k)
            k += 1
          }
          j += 1
        }
        if (k > 0) {
          cb += out.writeByte(b.toB)
          n += 1
        }
      }

      assert(n == nMissingBytes)
    }

    // Write fields
    fields.foreach { ef =>
      pv.loadField(cb, ef.name).consume(cb, { /* do nothing */ }, { pc =>
        val encodeField = ef.typ.buildEncoder(pc.pt, cb.emb.ecb)
        cb += encodeField(pc.code, out)
      })
    }
  }

  override def _buildFundamentalDecoder(
    pt: PType,
    mb: EmitMethodBuilder[_],
    region: Value[Region],
    in: Value[InputBuffer]
  ): Code[Long] = {
    val addr = mb.newLocal[Long]("addr")

    val pt2 = if (pt.isInstanceOf[PNDArray]) pt.asInstanceOf[PNDArray].representation else pt

    Code(
      addr := pt2.asInstanceOf[PBaseStruct].allocate(region),
      _buildInplaceDecoder(pt2, mb, region, addr, in),
      addr.load()
    )
  }

  override def _buildInplaceDecoder(
    pt: PType,
    mb: EmitMethodBuilder[_],
    region: Value[Region],
    addr: Value[Long],
    in: Value[InputBuffer]
  ): Code[Unit] = {

    val pt2 = if (pt.isInstanceOf[PNDArray]) pt.asInstanceOf[PNDArray].representation else pt
    val t = pt2.asInstanceOf[PBaseStruct]
    val mbytes = mb.newLocal[Long]("mbytes")

    val readFields = coerce[Unit](Code(fields.grouped(64).zipWithIndex.map { case (fieldGroup, groupIdx) =>
      val groupMB = mb.genEmitMethod(s"read_fields_group_$groupIdx", FastIndexedSeq[ParamType](classInfo[Region], LongInfo, LongInfo, classInfo[InputBuffer]), UnitInfo)
      val regionArg = groupMB.getCodeParam[Region](1)
      val addrArg = groupMB.getCodeParam[Long](2)
      val mbytesArg = groupMB.getCodeParam[Long](3)
      val inArg = groupMB.getCodeParam[InputBuffer](4)
      groupMB.emit(Code(fieldGroup.map { f =>
        if (t.hasField(f.name)) {
          val rf = t.field(f.name)
          val readElemF = f.typ.buildInplaceDecoder(rf.typ, mb.ecb)
          val rFieldAddr = t.fieldOffset(addrArg, rf.index)
          if (f.typ.required) {
            var c = readElemF(regionArg, rFieldAddr, inArg)
            if (!rf.typ.required) {
              c = Code(t.setFieldPresent(addrArg, rf.index), c)
            }
            c
          } else {
            Region.loadBit(mbytesArg, const(missingIdx(f.index).toLong)).mux(
              t.setFieldMissing(addrArg, rf.index),
              Code(
                t.setFieldPresent(addrArg, rf.index),
                readElemF(regionArg, rFieldAddr, inArg)))
          }
        } else {
          val skip = f.typ.buildSkip(groupMB)
          if (f.typ.required)
            skip(regionArg, inArg)
          else
            Region.loadBit(mbytesArg, const(missingIdx(f.index).toLong)).mux(
              Code._empty,
              skip(regionArg, inArg))
        }
      }))
      groupMB.invokeCode[Unit](region, addr, mbytes, in)
    }.toArray))

    Code(
      mbytes := region.allocate(const(1), const(nMissingBytes)),
      in.readBytes(region, mbytes, nMissingBytes),
      readFields,
      Code._empty)
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit = {
    val mbytes = cb.newLocal[Long]("mbytes", r.allocate(const(1), const(nMissingBytes)))
    cb += in.readBytes(r, mbytes, nMissingBytes)
    fields.foreach { f =>
      val skip = f.typ.buildSkip(cb.emb)
      if (f.typ.required)
        cb += skip(r, in)
      else
        cb.ifx(!Region.loadBit(mbytes, missingIdx(f.index).toLong), cb += skip(r, in))
    }
  }

  def _asIdent: String = {
    val sb = new StringBuilder
    sb.append("struct_of_")
    types.foreachBetween { ty =>
      sb.append(ty.asIdent)
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
      sb.append("EBaseStruct{")
      fields.foreachBetween(_.pretty(sb, indent, compact))(sb += ',')
      sb += '}'
    } else {
      if (fields.length == 0)
        sb.append("EBaseStruct { }")
      else {
        sb.append("EBaseStruct {")
        sb += '\n'
        fields.foreachBetween(_.pretty(sb, indent + 4, compact))(sb.append(",\n"))
        sb += '\n'
        sb.append(" " * indent)
        sb += '}'
      }
    }
  }

  def setRequired(newRequired: Boolean): EBaseStruct = EBaseStruct(fields, newRequired)
}
