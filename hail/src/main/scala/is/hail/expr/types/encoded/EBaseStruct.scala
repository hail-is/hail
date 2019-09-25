package is.hail.expr.types.encoded

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.types.BaseType
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual._
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.utils._

abstract class EBaseStruct extends EType {
  def types: Array[EType]
  def fields: IndexedSeq[EField]
  def size: Int = types.length
  def missingIdx: Array[Int]
  def nMissing: Int
  def nMissingBytes: Int

  def _buildDecoder(
    pt: PType,
    mb: MethodBuilder,
    region: Code[Region],
    in: Code[InputBuffer]
  ): Code[Long] = {
    val addr = mb.newLocal[Long]("addr")

    Code(
      addr := pt.asInstanceOf[PBaseStruct].allocate(region),
      _buildInplaceDecoder(pt, mb, region, addr, in),
      addr.load()
    )
  }

  def _buildSkip(mb: MethodBuilder, r: Code[Region], in: Code[InputBuffer]): Code[Unit] = {
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
      Code(skipFields: _*),
      Code._empty)
  }

  def asIdent: String = {
    val sb = new StringBuilder
    this match {
      case t: EStruct => sb.append("struct_of_")
      case t: ETuple => sb.append("tuple_of_")
    }
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
}
