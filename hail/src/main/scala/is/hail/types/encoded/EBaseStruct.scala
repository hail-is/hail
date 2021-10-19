package is.hail.types.encoded

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.BaseStruct
import is.hail.types.physical._
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.types.physical.stypes.concrete._
import is.hail.types.physical.stypes.interfaces.{SBaseStructValue, SLocus, SLocusValue}
import is.hail.types.virtual._
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

final case class EBaseStruct(fields: IndexedSeq[EField], override val required: Boolean = false) extends EType {
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
    fatal(s"cannot create struct with duplicate ${ plural(duplicates.size, "field") }: " +
      s"${ fieldNames.map(prettyIdentifier).mkString(", ") }", fieldNames.duplicates())
  }

  def _decodedSType(requestedType: Type): SType = requestedType match {
    case t: TInterval =>
      val structPType = decodedPType(t.structRepresentation).asInstanceOf[PStruct]
      val pointType = structPType.field("start").typ
      SIntervalPointer(PCanonicalInterval(pointType, false))
    case t: TLocus =>
      SCanonicalLocusPointer(PCanonicalLocus(t.rg, false))
    case t: TStruct =>
      val pFields = t.fields.map { case Field(name, typ, idx) =>
        val pt = fieldType(name).decodedPType(typ)
        PField(name, pt, idx)
      }
      SBaseStructPointer(PCanonicalStruct(pFields, false))
    case t: TTuple =>
      val pFields = t.fields.map { case Field(name, typ, idx) =>
        val pt = fieldType(name).decodedPType(typ)
        PTupleField(t._types(idx).index, pt)
      }
      SBaseStructPointer(PCanonicalTuple(pFields, false))
  }

  override def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit = {
    val structValue = v.st match {
      case SIntervalPointer(t: PCanonicalInterval) => new SBaseStructPointerValue(
        SBaseStructPointer(t.representation),
        v.asInstanceOf[SIntervalPointerValue].a)
      case _: SLocus => v.asInstanceOf[SLocusValue].structRepr(cb)
      case _ => v.asInstanceOf[SBaseStructValue]
    }
    // write missing bytes
    structValue.st match {
      case SBaseStructPointer(st) if st.size == size && st.fieldRequired.sameElements(fields.map(_.typ.required)) =>
        val missingBytes = UnsafeUtils.packBitsToBytes(st.nMissing)

        val addr = structValue.asInstanceOf[SBaseStructPointerValue].a
        if (nMissingBytes > 1)
          cb += out.writeBytes(addr, missingBytes - 1)
        if (nMissingBytes > 0)
          cb += out.writeByte((Region.loadByte(addr + (missingBytes.toLong - 1)).toI & const(EType.lowBitMask(st.nMissing & 0x7))).toB)

      case _ =>
        var j = 0
        var n = 0
        while (j < size) {
          var b: Code[Int] = 0
          var k = 0
          while (k < 8 && j < size) {
            val f = fields(j)
            if (!f.typ.required) {
              b = b | (structValue.isFieldMissing(cb, f.name).toI << k)
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
      structValue.loadField(cb, ef.name).consume(cb,
        {
          if (ef.typ.required)
            cb._fatal(s"required field ${ ef.name } saw missing value in encode")
        },
        { pc =>
          ef.typ.buildEncoder(pc.st, cb.emb.ecb)
            .apply(cb, pc.get, out)
        })
    }
  }

  override def _buildDecoder(cb: EmitCodeBuilder, t: Type, region: Value[Region], in: Value[InputBuffer]): SCode = {
    val pt = decodedPType(t)
    val addr = cb.newLocal[Long]("base_struct_dec_addr", region.allocate(pt.alignment, pt.byteSize))
    _buildInplaceDecoder(cb, pt, region, addr, in)
    pt.loadCheapSCode(cb, addr).get
  }

  override def _buildInplaceDecoder(cb: EmitCodeBuilder, pt: PType, region: Value[Region], addr: Value[Long], in: Value[InputBuffer]): Unit = {
    val structType: PCanonicalBaseStruct = pt match {
      case t: PCanonicalLocus => t.representation
      case t: PCanonicalInterval => t.representation
      case t: PCanonicalBaseStruct => t
    }
    val eStructMbytes = cb.newLocal[Long]("mbytes", region.allocate(const(1L), const(nMissingBytes)))
    cb += in.readBytes(region, eStructMbytes, nMissingBytes)

    val missingByte = cb.newLocal[Int]("ebase_struct_decode_missing_byte")
    var currentMissingBitIdx = 0
    var currentMissingByteIdx = 0L

    fields.foreach { ef =>
      var bytesToShift = 0
      if (structType.hasField(ef.name)) {
        val rf = structType.field(ef.name)
        val readElemF = ef.typ.buildInplaceDecoder(rf.typ, cb.emb.ecb)
        val rFieldAddr = structType.fieldOffset(addr, rf.index)
        if (ef.typ.required) {
          readElemF(cb, region, rFieldAddr, in)
          if (!rf.typ.required) {
            bytesToShift = 1
            currentMissingBitIdx += 1
          }
        } else {
          cb.ifx(Region.loadBit(eStructMbytes, const(missingIdx(ef.index).toLong)), {
            //cb.println(s"SET MISSING BYTE TO NONZERO VALUE for field ${ef}")
            cb.assign(missingByte, missingByte + 0x80)
          }, {
            readElemF(cb, region, rFieldAddr, in)
          })
          bytesToShift = 1
          currentMissingBitIdx += 1
        }
      } else {
        val skip = ef.typ.buildSkip(cb.emb.ecb)
        if (ef.typ.required)
          skip(cb, region, in)
        else
          cb.ifx(!Region.loadBit(eStructMbytes, const(missingIdx(ef.index).toLong)), skip(cb, region, in))
      }

      if (currentMissingBitIdx == 8) {
        structType.setMissingByte(cb, addr, currentMissingByteIdx, missingByte.toB)
        cb.assign(missingByte, 0)
        currentMissingBitIdx = 0
        currentMissingByteIdx += 1L
      } else {
        cb.assign(missingByte, missingByte >>> bytesToShift)
      }
    }
    // Fix up last missing byte
    if (currentMissingBitIdx != 0) {
      cb.assign(missingByte, missingByte >>> (const(8) - currentMissingBitIdx - 1))
      structType.setMissingByte(cb, addr, structType.nMissingBytes - 1, missingByte.toB)
    }
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit = {
    val mbytes = cb.newLocal[Long]("mbytes", r.allocate(const(1L), const(nMissingBytes)))
    cb += in.readBytes(r, mbytes, nMissingBytes)
    fields.foreach { f =>
      val skip = f.typ.buildSkip(cb.emb.ecb)
      if (f.typ.required)
        skip(cb, r, in)
      else
        cb.ifx(!Region.loadBit(mbytes, missingIdx(f.index).toLong), skip(cb, r, in))
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
