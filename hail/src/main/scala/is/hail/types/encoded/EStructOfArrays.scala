package is.hail.types.encoded

import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s._
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.PBaseStruct
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SStructOfArrays, SStructOfArraysSettable}
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.types.virtual.{Field, TArray, TBaseStruct, Type}
import is.hail.utils._
import is.hail.types.physical.stypes.interfaces.SContainer

case class EStructOfArrays(fields: IndexedSeq[EField], elementsRequired: Boolean, override val required: Boolean = false) extends EType {
  assert(fields.zipWithIndex.forall { case (f, i) => f.index == i  && f.isInstanceOf[EContainer]})

  val types: Array[EContainer] = fields.map(_.typ.asInstanceOf).toArray
  val fieldIdx: Map[String, Int] = fields.map(f => f.name -> f.index).toMap
  def fieldType(name: String): EContainer = types(fieldIdx(name))

  def _decodedSType(requestedType: Type): SType = {
    requestedType match {
      case ta @ TArray(t: TBaseStruct) =>
        SStructOfArrays(ta, elementsRequired, t.fields.map { case Field(name, typ, _) =>
          fieldType(name).decodedSType(TArray(typ)).asInstanceOf[SContainer]
        })
    }
  }

  def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit = {
    v.st match {
      case SIndexablePointer(pc) if pc.elementType.isInstanceOf[PBaseStruct] =>
        throw new NotImplementedError
      case _: SStructOfArrays =>
        buildSimpleEncoder(cb, v.asInstanceOf[SStructOfArraysSettable], out)
    }
  }

  private def buildSimpleEncoder(cb: EmitCodeBuilder, v: SStructOfArraysSettable, out: Value[OutputBuffer]): Unit = {
    assert(v.lookupOrLength.isRight == elementsRequired)
    cb += out.writeInt(v.loadLength())

    v.lookupOrLength match {
      case Left(lookup) =>
        val region = cb.emb.partitionRegion
        val nbytes = cb.newLocal("n_missing_bytes", UnsafeUtils.packBitsToBytes(v.loadLength()))
        val mbytes = region.allocate(const(1L), nbytes.toL)
        cb += Region.setMemory(mbytes, nbytes.toL, const(0).toB)
        lookup.forEachDefined(cb) { (cb, i, lv) =>
          cb.ifx(lv.asInt32.intCode(cb).ceq(SStructOfArrays.MISSING_SENTINEL), {
            cb += Region.setBit(mbytes, i.toL)
          })
        }
        cb += out.writeBytes(mbytes, nbytes)
    }

    for ((field, etype) <- v.fields.zip(types)) {
      etype.buildEncoder(field.st, cb.emb.ecb).apply(cb, field, out)
    }
  }

  def _buildDecoder(cb: EmitCodeBuilder, t: Type, region: Value[Region], in: Value[InputBuffer]): SCode = {
    val st = decodedSType(t).asInstanceOf[SStructOfArrays]
    val structType: TBaseStruct = t match {
      case TArray(t: TBaseStruct) => t
    }

    val result = SStructOfArraysSettable(cb.localBuilder, st, "decode_result")
    result.lookupOrLength match {
      case Left(lookup) => throw new NotImplementedError // TODO
      case Right(length) => cb.assign(length, in.readInt())
    }
    fields.foreach { case EField(name, typ, index) =>
      if (structType.hasField(name)) {
        val rf = structType.field(name)
        val readFieldF = typ.buildDecoder(TArray(rf.typ), cb.emb.ecb)
        result.fields(rf.index).store(cb, readFieldF.apply(cb, region, in))
      } else {
        typ.buildSkip(cb.emb).apply(region, in)
      }
    }

    result
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit = ???

  def _asIdent: String = {
    val sb = new StringBuilder
    sb.append("structofarrays_of_")
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
    _pretty(sb, indent = 0, compact = true)
    sb.result()
  }

  override def _pretty(sb: StringBuilder, indent: Int, compact: Boolean): Unit = {
    val elementsRequiredStr = if (elementsRequired) "True" else "False"
    if (compact) {
      sb.append(s"EStructOfArrays[$elementsRequiredStr]{")
      fields.foreachBetween(_.pretty(sb, indent, compact))(sb += ',')
    } else {
      if (fields.isEmpty) {
        sb.append(s"EStructOfArrays[$elementsRequiredStr] { }")
      } else {
        sb.append(s"EStructOfArrays[$elementsRequiredStr] {")
        sb += '\n'
        fields.foreachBetween(_.pretty(sb, indent + 4, compact))(sb.append(",\n"))
        sb += '\n'
        sb.append(" " * indent)
        sb += '}'
      }
    }
  }

  def setRequired(required: Boolean): EType = if (required == this.required) this else EStructOfArrays(fields, elementsRequired, required)
}
