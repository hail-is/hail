package is.hail.expr.types.encoded

import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s._
import is.hail.expr.ir.typeToTypeInfo
import is.hail.expr.types.BaseType
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual.Type
import is.hail.io.{InputBuffer, OutputBuffer}

// All _$methods here assume that their arguments are fundamental types
abstract class EType extends BaseType with Serializable {
  def virtualType: Type

  type StagedEncoder = (Code[_], Code[OutputBuffer]) => Code[Unit]
  type StagedDecoder[T] = (Code[Region], Code[InputBuffer]) => Code[T]
  type StagedInplaceDecoder = (Code[Region], Code[Long], Code[InputBuffer]) => Code[Unit]

  final def buildEncoder(pt: PType, methodBuilder: MethodBuilder): StagedEncoder = {
    val mb = buildEncoderMethod(pt, methodBuilder.fb)
    mb.invoke(_, _)
  }

  final def buildEncoderMethod(pt: PType, fb: FunctionBuilder[_]): MethodBuilder = {
    require(encodeCompatible(pt))
    val ptti = typeToTypeInfo(pt)
    val mb = fb.newMethod(s"ENCODE_${pt.asIdent}_TO_${asIdent}",
      Array[TypeInfo[_]](ptti, classInfo[OutputBuffer]),
      UnitInfo)
    val arg = mb.getArg(1)(ptti)
    val out: Code[OutputBuffer] = mb.getArg[OutputBuffer](2)
    mb.emit(_buildEncoder(pt.fundamentalType, mb, arg, out))
    mb
  }

  final def buildDecoder[T](pt: PType, methodBuilder: MethodBuilder): StagedDecoder[T] = {
    val mb = buildDecoderMethod(pt, methodBuilder.fb)
    mb.invoke(_, _)
  }

  final def buildDecoderMethod[T](pt: PType, fb: FunctionBuilder[_]): MethodBuilder = {
    require(decodeCompatible(pt))
    val mb = fb.newMethod(s"DECODE_${asIdent}_TO_${pt.asIdent}",
      Array[TypeInfo[_]](typeInfo[Region], classInfo[InputBuffer]),
      typeToTypeInfo(pt))
    val region: Code[Region] = mb.getArg[Region](1)
    val in: Code[InputBuffer] = mb.getArg[InputBuffer](2)
    val dec = _buildDecoder(pt.fundamentalType, mb, region, in)
    mb.emit(dec)
    mb
  }

  final def buildInplaceDecoder(pt: PType, methodBuilder: MethodBuilder): StagedInplaceDecoder = {
    require(decodeCompatible(pt))
    val me = methodBuilder.fb.newMethod(s"INPLACE_DECODE_${asIdent}_TO_${pt.asIdent}",
      Array[TypeInfo[_]](typeInfo[Region], typeInfo[Long], classInfo[InputBuffer]),
      UnitInfo)
    val region: Code[Region] = me.getArg[Region](1)
    val addr: Code[Long] = me.getArg[Long](2)
    val in: Code[InputBuffer] = me.getArg[InputBuffer](3)
    val dec = _buildInplaceDecoder(pt.fundamentalType, me, region, addr, in)
    me.emit(dec)
    me.invoke(_, _, _)
  }

  final def buildSkip(methodBuilder: MethodBuilder): (Code[Region], Code[InputBuffer]) => Code[Unit] = {
    val me = methodBuilder.fb.newMethod(s"SKIP_${asIdent}",
      Array[TypeInfo[_]](classInfo[Region], classInfo[InputBuffer]),
      UnitInfo)
    val r: Code[Region] = me.getArg[Region](1)
    val in: Code[InputBuffer] = me.getArg[InputBuffer](2)
    val skip = _buildSkip(me, r, in)
    me.emit(skip)
    me.invoke(_, _)
  }

  def _buildEncoder(pt: PType, mb: MethodBuilder, v: Code[_], out: Code[OutputBuffer]): Code[Unit]
  def _buildDecoder(pt: PType, mb: MethodBuilder, region: Code[Region], in: Code[InputBuffer]): Code[_]
  def _buildInplaceDecoder(
    pt: PType,
    mb: MethodBuilder,
    region: Code[Region],
    addr: Code[Long],
    in: Code[InputBuffer]
  ): Code[_] = {
    assert(!pt.isInstanceOf[PBaseStruct]) // should be overridden for structs
    val decoded = _buildDecoder(pt, mb, region, in)
    Region.storeIRIntermediate(pt)(addr, decoded)
  }

  def _buildSkip(mb: MethodBuilder, r: Code[Region], in: Code[InputBuffer]): Code[Unit]

  def _compatible(pt: PType): Boolean = ???
  // Can this etype encode from this ptype
  final def encodeCompatible(pt: PType): Boolean = _encodeCompatible(pt.fundamentalType)
  def _encodeCompatible(pt: PType): Boolean = _compatible(pt)
  // Can this etype decode to this ptype
  final def decodeCompatible(pt: PType): Boolean = _decodeCompatible(pt.fundamentalType)
  def _decodeCompatible(pt: PType): Boolean = _compatible(pt)

  def required: Boolean

  final def pretty(sb: StringBuilder, indent: Int, compact: Boolean) {
    if (required)
      sb.append("+")
    _pretty(sb, indent, compact)
  }

  def asIdent: String

  def _toPretty: String

  def _pretty(sb: StringBuilder, indent: Int, compact: Boolean) {
    sb.append(_toPretty)
  }

  final def decodedPType(requestedType: Type): PType = {
    val ret: PType = _decodedPType(requestedType)

    assert(decodeCompatible(ret),
      s"""Invalid requested type, cannot decode
         |encoded type  : ${this}
         |requested type: $requestedType""".stripMargin)
    ret
  }
  def _decodedPType(requestedType: Type): PType
}

object EType {
  // The 'entry point' for building an encoder from an EType and a PType
  def buildEncoder(et: EType, pt: PType): () => AsmFunction2[Long, OutputBuffer, Unit] = {
    val fb = new Function2Builder[Long, OutputBuffer, Unit]("etypeEncode")
    val mb = fb.apply_method
    val f = et.buildEncoder(pt, mb)

    val addr: Code[Long] = mb.getArg[Long](1)
    val out: Code[OutputBuffer] = mb.getArg[OutputBuffer](2)
    // XXX get or load?
    val v = Region.getIRIntermediate(pt)(addr)

    mb.emit(f(v, out))
    fb.result()
  }

  def buildDecoder(et: EType, t: Type): (PType, () => AsmFunction2[Region, InputBuffer, Long]) = {
    val fb = new Function2Builder[Region, InputBuffer, Long]("etypeDecode")
    val mb = fb.apply_method
    val pt = et.decodedPType(t)
    val f = et.buildDecoder(pt, mb)

    val region: Code[Region] = mb.getArg[Region](1)
    val in: Code[InputBuffer] = mb.getArg[InputBuffer](2)

    if (pt.isPrimitive) {
      val srvb = new StagedRegionValueBuilder(mb, pt)
      mb.emit(Code(
        srvb.start(),
        srvb.addIRIntermediate(pt)(f(region, in)),
        srvb.end()))
    } else {
      mb.emit(f(region, in))
    }

    (pt, fb.result())
  }

  def defaultFromPType(pt: PType): EType = defaultFromPType(pt, pt.required)
  def defaultFromPType(pt: PType, required: Boolean): EType = {
    pt.fundamentalType match {
      case t: PInt32 => EInt32(t.required)
      case t: PInt64 => EInt64(t.required)
      case t: PFloat32 => EFloat32(t.required)
      case t: PFloat64 => EFloat64(t.required)
      case t: PBoolean => EBoolean(t.required)
      case t: PBinary => EBinary(t.required)
      case t: PArray => EArray(defaultFromPType(t.elementType), t.required)
      case t: PTuple => ETuple(t._types.map(pf => ETupleField(pf.index, defaultFromPType(pf.typ))), t.required)
      case t: PStruct => EStruct(t.fields.map(f => EField(f.name, defaultFromPType(f.typ), f.index)), t.required)
    }
  }
}
