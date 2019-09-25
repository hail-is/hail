package is.hail.expr.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.types.{BaseStruct, BaseType}
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
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

final case class EStruct(fields: IndexedSeq[EField], override val required: Boolean = false) extends EBaseStruct {
  lazy val virtualType: TStruct = TStruct(fields.map(f => Field(f.name, f.typ.virtualType, f.index)), required)

  assert(fields.zipWithIndex.forall { case (f, i) => f.index == i })

  val types: Array[EType] = fields.map(_.typ).toArray
  val fieldNames: Array[String] = fields.map(_.name).toArray
  val fieldIdx: Map[String, Int] = fields.map(f => (f.name, f.index)).toMap
  def hasField(name: String): Boolean = fieldIdx.contains(name)
  def fieldType(name: String): EType = types(fieldIdx(name))
  val missingIdx = new Array[Int](size)
  val nMissing: Int = BaseStruct.getMissingness[EType](types, missingIdx)
  val nMissingBytes = (nMissing + 7) >>> 3

  if (!fieldNames.areDistinct()) {
    val duplicates = fieldNames.duplicates()
    fatal(s"cannot create struct with duplicate ${plural(duplicates.size, "field")}: " +
      s"${fieldNames.map(prettyIdentifier).mkString(", ")}", fieldNames.duplicates())
  }

  override def _decodeCompatible(pt: PType): Boolean = {
    if (!pt.isInstanceOf[PStruct])
      false
    else {
      val ps = pt.asInstanceOf[PStruct]
      ps.required == required &&
        ps.size <= size &&
        ps.fields.forall { f =>
          hasField(f.name) && fieldType(f.name).decodeCompatible(f.typ)
        }
    }
  }

  override def _encodeCompatible(pt: PType): Boolean = {
    if (!pt.isInstanceOf[PStruct])
      false
    else {
      val ps = pt.asInstanceOf[PStruct]
      ps.required == required &&
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
      PInterval(pointType, required)
    case t: TLocus => PLocus(t.rg, required)
    case t: TStruct =>
      val pFields = t.fields.map { case Field(name, typ, idx) =>
        val pt = fieldType(name).decodedPType(typ)
        PField(name, pt, idx)
      }
      PStruct(pFields, required)
  }

  override def _buildEncoder(pt: PType, mb: MethodBuilder, v: Code[_], out: Code[OutputBuffer]): Code[Unit] = {
    val ft = pt.asInstanceOf[PStruct]
    val addr = coerce[Long](v)
    val writeMissingBytes = if (ft.size == size) {
      out.writeBytes(addr, ft.nMissingBytes)
    } else {
      var c: Code[Unit] = Code._empty[Unit]
      var j = 0
      var n = 0
      while (j < size) {
        var b = const(0)
        var k = 0
        while (k < 8 && j < size) {
          val f = fields(j)
          if (!f.typ.required) {
            val i = ft.fieldIdx(f.name)
            b = b | (ft.isFieldMissing(addr, i).toI << k)
            k += 1
          }
          j += 1
        }
        if (k > 0) {
          c = Code(c, out.writeByte(b.toB))
          n += 1
        }
      }
      assert(n == nMissingBytes)
      c
    }

    val writeFields = Code(fields.map { ef =>
      val i = ft.fieldIdx(ef.name)
      val pf = ft.fields(i)
      val encodeField = ef.typ.buildEncoder(pf.typ, mb)
      val v = Region.loadIRIntermediate(pf.typ)(ft.fieldOffset(addr, i))
      ft.isFieldDefined(addr, i).mux(
        encodeField(v, out),
        Code._empty[Unit]
      )
    }: _*)

    Code(writeMissingBytes, writeFields, Code._empty[Unit])
  }

  override def _buildInplaceDecoder(
    pt: PType,
    mb: MethodBuilder,
    region: Code[Region],
    addr: Code[Long],
    in: Code[InputBuffer]
  ): Code[Unit] = {
    val mbytes = mb.newLocal[Long]("mbytes")

    val t = pt.asInstanceOf[PStruct]
    val readFields = fields.map { f =>
      if (t.hasField(f.name)) {
        val rf = t.field(f.name)
        val readElemF = f.typ.buildInplaceDecoder(rf.typ, mb)
        val rFieldAddr = t.fieldOffset(addr, rf.index)
        if (f.typ.required)
          readElemF(region, rFieldAddr, in)
        else
          Region.loadBit(mbytes, const(missingIdx(f.index).toLong)).mux(
            t.setFieldMissing(addr, rf.index),
            Code(
              t.setFieldPresent(addr, rf.index),
              readElemF(region, rFieldAddr, in)))
      } else {
        val skip = f.typ.buildSkip(mb)
        if (f.typ.required)
          skip(region, in)
        else
          Region.loadBit(mbytes, const(missingIdx(f.index).toLong)).mux(
            Code._empty[Unit],
            skip(region, in))
      }
    }

    Code(
      mbytes := region.allocate(const(1), const(nMissingBytes)),
      in.readBytes(region, mbytes, nMissingBytes),
      Code(readFields: _*),
      Code._empty[Unit])
  }

  def identBase: String = "struct"
  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean) {
    if (compact) {
      sb.append("Struct{")
      fields.foreachBetween(_.pretty(sb, indent, compact))(sb += ',')
      sb += '}'
    } else {
      if (fields.length == 0)
        sb.append("Struct { }")
      else {
        sb.append("Struct {")
        sb += '\n'
        fields.foreachBetween(_.pretty(sb, indent + 4, compact))(sb.append(",\n"))
        sb += '\n'
        sb.append(" " * indent)
        sb += '}'
      }
    }
  }
}
