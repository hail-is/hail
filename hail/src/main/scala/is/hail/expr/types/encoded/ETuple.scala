package is.hail.expr.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.types.{BaseStruct, BaseType}
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

case class ETupleField(index: Int, typ: EType)

final case class ETuple(_types: IndexedSeq[ETupleField], override val required: Boolean = false) extends EBaseStruct {
  lazy val virtualType: TTuple = TTuple(_types.map(tf => TupleField(tf.index, tf.typ.virtualType)), required)
  val types = _types.map(_.typ).toArray
  val fields: IndexedSeq[EField] = types.zipWithIndex.map { case (t, i) => EField(s"$i", t, i) }
  val missingIdx = new Array[Int](size)
  val nMissing: Int = BaseStruct.getMissingness[EType](types, missingIdx)
  val nMissingBytes = (nMissing + 7) >>> 3

  override def _decodeCompatible(pt: PType): Boolean = {
    pt.required == required && pt.isInstanceOf[PTuple] &&
      pt.asInstanceOf[PTuple].size <= size &&
      pt.asInstanceOf[PTuple].fields.forall { f =>
        types(f.index).decodeCompatible(f.typ)
      }
  }

  override def _encodeCompatible(pt: PType): Boolean = {
    pt.required == required &&
      pt.isInstanceOf[PTuple] &&
      size <= pt.asInstanceOf[PTuple].size &&
      pt.asInstanceOf[PTuple].types.zip(types).forall {
        case (pt, et) => et.encodeCompatible(pt)
      }
  }

  def _decodedPType(requestedType: Type): PType = {
    val t = requestedType.asInstanceOf[TTuple]
    val pFields = t.fields.map { case Field(_, typ, idx) =>
      val pt = types(idx).decodedPType(typ)
      PTupleField(idx, pt)
    }
    PTuple(pFields, required)
  }

  def _buildEncoder(pt: PType, mb: MethodBuilder, v: Code[_], out: Code[OutputBuffer]): Code[Unit] = {
    // if we are here, then then we are a like a tuple, and the etype is a prefix of the
    // ptype.
    val ft = pt.asInstanceOf[PBaseStruct]
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
          val et = types(j)
          if (!et.required) {
            b = b | (ft.isFieldMissing(addr, j).toI << k)
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

    val fieldEncoders = types.zipWithIndex.map { case (et, i) => (et.buildEncoder(ft.types(i), mb), i) }
    val writeFields = Code(fieldEncoders.map {
      case (encodeField, i) => ft.isFieldDefined(addr, i).mux(
        encodeField(Region.loadIRIntermediate(ft.types(i))(ft.fieldOffset(addr, i)), out),
        Code._empty[Unit])
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

    val t = pt.asInstanceOf[PTuple]
    val readFields = new Array[Code[_]](size)

    var i = 0
    var j = 0
    while (i < size) {
      val f = _types(i)
      readFields(i) =
        if (t._types(j).index == f.index) {
          val rf = t._types(j)
          j += 1

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

      i += 1
    }
    assert(j == t.size)

    Code(
      mbytes := region.allocate(const(1), const(nMissingBytes)),
      in.readBytes(region, mbytes, nMissingBytes),
      Code(readFields: _*),
      Code._empty[Unit])
  }

  def identBase: String = "tuple"
  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean) {
    sb.append("Tuple[")
    types.foreachBetween(_.pretty(sb, indent, compact))(sb += ',')
    sb += ']'
  }
}
