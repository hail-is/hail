package is.hail.types.encoded

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir.{EmitMethodBuilder, ParamType}
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
    if (!pt.isInstanceOf[PBaseStruct])
      false
    else {
      val ps = pt.asInstanceOf[PBaseStruct]
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

  override def _buildFundamentalEncoder(pt: PType, mb: EmitMethodBuilder[_], v: Value[_], out: Value[OutputBuffer]): Code[Unit] = {
    val ft = pt.asInstanceOf[PBaseStruct]
    val vs = coerce[Long](v)
    val writeMissingBytes = if (ft.size == size) {
      val missingBytes = UnsafeUtils.packBitsToBytes(ft.nMissing)
      var c = Code._empty
      ft match {
        case ps: PCanonicalBaseStruct if ps.fieldRequired.sameElements(fields.map(_.typ.required)) =>
          if (nMissingBytes > 1)
            c = Code(c, out.writeBytes(coerce[Long](v), missingBytes - 1))
          if (nMissingBytes > 0)
            c = Code(c, out.writeByte((Region.loadByte(vs + (missingBytes.toLong - 1)).toI & const(EType.lowBitMask(ft.nMissing & 0x7))).toB))
        case _ =>
          fields.filter(f => !f.typ.required)
            .grouped(8)
            .foreach { group =>
              c = Code(c, out.writeByte(group.zipWithIndex.map { case (f, i) =>
                ft.isFieldMissing(vs, f.index).toI << i
              }.reduce(_ | _).toB))
            }
      }
      c
    } else {
      val groupSize = 64
      var methodIdx = 0
      var currentMB = mb.genEmitMethod(s"missingbits_group_$methodIdx", FastIndexedSeq[ParamType](LongInfo, classInfo[OutputBuffer]), UnitInfo)
      var wrappedC: Code[Unit] = Code._empty
      var methodC: Code[Unit] = Code._empty

      var j = 0
      var n = 0
      while (j < size) {
        if (n % groupSize == 0) {
          currentMB.emit(methodC)
          methodC = Code._empty
          wrappedC = Code(wrappedC, currentMB.invokeCode[Unit](v, out))
          methodIdx += 1
          currentMB = mb.genEmitMethod(s"missingbits_group_$methodIdx", FastIndexedSeq[ParamType](LongInfo, classInfo[OutputBuffer]), UnitInfo)
        }
        var b: Code[Int] = 0
        var k = 0
        while (k < 8 && j < size) {
          val f = fields(j)
          if (!f.typ.required) {
            val i = ft.fieldIdx(f.name)
            b = b | (ft.isFieldMissing(currentMB.getCodeParam[Long](1), i).toI << k)
            k += 1
          }
          j += 1
        }
        if (k > 0) {
          methodC = Code(methodC, currentMB.getCodeParam[OutputBuffer](2).writeByte(b.toB))
          n += 1
        }
      }
      currentMB.emit(methodC)
      wrappedC = Code(wrappedC, currentMB.invokeCode[Unit](v, out))

      assert(n == nMissingBytes)
      wrappedC
    }

    val writeFields = Code(fields.grouped(64).zipWithIndex.map { case (fieldGroup, groupIdx) =>
      val groupMB = mb.genEmitMethod(s"write_fields_group_$groupIdx", FastIndexedSeq[ParamType](LongInfo, classInfo[OutputBuffer]), UnitInfo)

      val addr = groupMB.getCodeParam[Long](1)
      val out2 = groupMB.getCodeParam[OutputBuffer](2)
      groupMB.emit(Code(
        fieldGroup.map { ef =>
          val i = ft.fieldIdx(ef.name)
          val pf = ft.fields(i)
          val encodeField = ef.typ.buildEncoder(pf.typ, groupMB.ecb)
          val v = Region.loadIRIntermediate(pf.typ)(ft.fieldOffset(addr, i))
          ft.isFieldDefined(addr, i).mux(
            encodeField(v, out2),
            Code._empty
          )
        }
      ))

      groupMB.invokeCode[Unit](v, out)
    }.toArray)

    Code(writeMissingBytes, writeFields, Code._empty)
  }

  override def _buildFundamentalDecoder(
    pt: PType,
    mb: EmitMethodBuilder[_],
    region: Value[Region],
    in: Value[InputBuffer]
  ): Code[Long] = {
    val addr = mb.newLocal[Long]("addr")

    Code(
      addr := pt.asInstanceOf[PBaseStruct].allocate(region),
      _buildInplaceDecoder(pt, mb, region, addr, in),
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

    val t = pt.asInstanceOf[PBaseStruct]
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
  def _buildSkip(mb: EmitMethodBuilder[_], r: Value[Region], in: Value[InputBuffer]): Code[Unit] = {
    val mbytes = mb.newLocal[Long]("mbytes")
    val skipFields = fields.map { f =>
      val skip = f.typ.buildSkip(mb)
      if (f.typ.required)
        skip(r, in)
      else
        Region.loadBit(mbytes, missingIdx(f.index).toLong).mux(
          Code._empty,
          skip(r, in))
    }

    Code(
      mbytes := r.allocate(const(1), const(nMissingBytes)),
      in.readBytes(r, mbytes, nMissingBytes),
      Code(skipFields))
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
