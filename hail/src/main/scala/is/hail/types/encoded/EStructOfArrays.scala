package is.hail.types.encoded
import is.hail.annotations.{Region, UnsafeUtils}
import is.hail.asm4s.Value
import is.hail.expr.ir.EmitCodeBuilder
import is.hail.io.{InputBuffer, OutputBuffer}
import is.hail.types.physical.PBaseStruct
import is.hail.types.physical.stypes.concrete.{SIndexablePointer, SStructOfArrays, SStructOfArraysSettable}
import is.hail.types.physical.stypes.interfaces.SIndexableValue
import is.hail.types.physical.stypes.{SCode, SType, SValue}
import is.hail.types.virtual.Type
import is.hail.utils._

case class EStructOfArrays(fields: IndexedSeq[EField], elementsRequired: Boolean, override val required: Boolean = false) extends EType {
  assert(fields.zipWithIndex.forall { case (f, i) => f.index == i  && f.isInstanceOf[EContainer]})

  val types: Array[EType] = fields.map(_.typ).toArray

  def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit = {
    // dumb version, bad skips
    v.st match {
      case SIndexablePointer(pc) if pc.elementType.isInstanceOf[PBaseStruct] =>
        throw new NotImplementedError
      case _: SStructOfArrays =>
        encodeFastPath(cb, v.asInstanceOf[SStructOfArraysSettable], out)
    }
  }

  private def encodeFastPath(cb: EmitCodeBuilder, v: SStructOfArraysSettable, out: Value[OutputBuffer]): Unit = {
    val structReq = v.st.elementsRequired
    cb += out.writeInt(v.loadLength())
    v.missing.foreach(m => cb += out.writeBytes(m, UnsafeUtils.packBitsToBytes(v.loadLength())))
    val i = cb.newLocal[Int]("i", 0)
    for ((sfield, etype) <- v.fields.zip(types)) {
      val fieldReq = sfield.st.elementEmitType.required
      if (!fieldReq || fieldReq == structReq) {
        etype.buildEncoder(sfield.st, cb.emb.ecb).apply(cb, sfield, out)
      } else {
        assert(fieldReq && !structReq) // element required, struct optional
        cb.forLoop(cb.assign(i, 0), i < v.loadLength(), cb.assign(i, i + 1), {
          cb.ifx(!v.isElementMissing(i), {
            etype.asInstanceOf[EContainer].elementType.buildEncoder(sfield.st.elementType, cb.emb.ecb)
              .apply(cb, sfield.loadElement(cb, i)
                .get(cb, "required value cannot be missing")
                .memoize(cb, "encode_field"), out)
          })
        })
      }
    }
  }

  def _buildDecoder(cb: EmitCodeBuilder, t: Type, region: Value[Region], in: Value[InputBuffer]): SCode = ???

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

  def _decodedSType(requestedType: Type): SType = ???

  def setRequired(required: Boolean): EType = if (required == this.required) this else EStructOfArrays(fields, elementsRequired, required)
}
