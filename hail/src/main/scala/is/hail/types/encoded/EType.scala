package is.hail.types.encoded
import java.util
import java.util.Map.Entry

import is.hail.HailContext
import is.hail.annotations.{Region, StagedRegionValueBuilder}
import is.hail.asm4s.{coerce => _, _}
import is.hail.expr.ir.{EmitClassBuilder, EmitFunctionBuilder, EmitMethodBuilder, ExecuteContext, IRParser, ParamType, PunctuationToken, TokenIterator}
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.types._
import is.hail.io._
import is.hail.utils._
import org.json4s.CustomSerializer
import org.json4s.JsonAST.JString


class ETypeSerializer extends CustomSerializer[EType](format => ( {
  case JString(s) => IRParser.parse[EType](s, EType.eTypeParser)
}, {
  case t: EType => JString(t.parsableString())
}))


abstract class EType extends BaseType with Serializable with Requiredness {
  type StagedEncoder = (Code[_], Code[OutputBuffer]) => Code[Unit]
  type StagedDecoder[T] = (Code[Region], Code[InputBuffer]) => Code[T]
  type StagedInplaceDecoder = (Code[Region], Code[Long], Code[InputBuffer]) => Code[Unit]

  final def buildEncoder(ctx: ExecuteContext, t: PType): (OutputBuffer) => Encoder = {
    val f = EType.buildEncoder(ctx, this, t)
    out: OutputBuffer => new CompiledEncoder(out, f)
  }

  final def buildDecoder(ctx: ExecuteContext, requestedType: Type): (PType, (InputBuffer) => Decoder) = {
    val (rt, f) = EType.buildDecoder(ctx, this, requestedType)
    (rt, (in: InputBuffer) => new CompiledDecoder(in, f))
  }

  final def buildStructDecoder(ctx: ExecuteContext, requestedType: TStruct): (PStruct, (InputBuffer) => Decoder) = {
    val (pType: PStruct, makeDec) = buildDecoder(ctx, requestedType)
    pType -> makeDec
  }

  final def buildEncoder(pt: PType, cb: EmitClassBuilder[_]): StagedEncoder = {
    buildEncoderMethod(pt, cb).invokeCode(_, _)
  }

  final def buildEncoderMethod(pt: PType, cb: EmitClassBuilder[_]): EmitMethodBuilder[_] = {
    if (!encodeCompatible(pt))
      throw new RuntimeException(s"encode incompatible:\n  PT: $pt\n  ET: ${ parsableString() }")
    val ptti = typeToTypeInfo(pt)
    cb.getOrGenEmitMethod(s"ENCODE_${ pt.asIdent }_TO_${ asIdent }",
      (pt, this, "ENCODE"),
      FastIndexedSeq[ParamType](ptti, classInfo[OutputBuffer]),
      UnitInfo) { mb =>

      val arg = mb.getCodeParam(1)(ptti)
      val out = mb.getCodeParam[OutputBuffer](2)
      mb.emit(_buildEncoder(pt, mb, arg, out))
    }
  }

  final def buildDecoder[T](pt: PType, cb: EmitClassBuilder[_]): StagedDecoder[T] = {
    buildDecoderMethod(pt, cb).invokeCode(_, _)
  }

  final def buildDecoderMethod[T](pt: PType, cb: EmitClassBuilder[_]): EmitMethodBuilder[_] = {
    if (!decodeCompatible(pt))
      throw new RuntimeException(s"decode incompatible:\n  PT: $pt }\n  ET: ${ parsableString() }")
    cb.getOrGenEmitMethod(s"DECODE_${ asIdent }_TO_${ pt.asIdent }",
      (pt, this, "DECODE"),
      FastIndexedSeq[ParamType](typeInfo[Region], classInfo[InputBuffer]),
      typeToTypeInfo(pt)) { mb =>

      val region: Value[Region] = mb.getCodeParam[Region](1)
      val in: Value[InputBuffer] = mb.getCodeParam[InputBuffer](2)
      val dec = _buildDecoder(pt, mb, region, in)
      mb.emit(dec)
    }
  }

  final def buildInplaceDecoder(pt: PType, cb: EmitClassBuilder[_]): StagedInplaceDecoder = {
    buildInplaceDecoderMethod(pt, cb).invokeCode(_, _, _)
  }

  final def buildInplaceDecoderMethod(pt: PType, cb: EmitClassBuilder[_]): EmitMethodBuilder[_] = {
    if (!decodeCompatible(pt))
      throw new RuntimeException(s"decode incompatible:\n  PT: $pt\n  ET: ${ parsableString() }")
    cb.getOrGenEmitMethod(s"INPLACE_DECODE_${ asIdent }_TO_${ pt.asIdent }",
      (pt, this, "INPLACE_DECODE"),
      FastIndexedSeq[ParamType](typeInfo[Region], typeInfo[Long], classInfo[InputBuffer]),
      UnitInfo)({ mb =>

      val region: Value[Region] = mb.getCodeParam[Region](1)
      val addr: Value[Long] = mb.getCodeParam[Long](2)
      val in: Value[InputBuffer] = mb.getCodeParam[InputBuffer](3)
      val dec = _buildInplaceDecoder(pt.encodableType, mb, region, addr, in)
      mb.emit(dec)
    })
  }

  final def buildSkip(mb: EmitMethodBuilder[_]): (Code[Region], Code[InputBuffer]) => Code[Unit] = {
    mb.getOrGenEmitMethod(s"SKIP_${ asIdent }",
      (this, "SKIP"),
      FastIndexedSeq[ParamType](classInfo[Region], classInfo[InputBuffer]),
      UnitInfo)({ mb =>

      val r: Value[Region] = mb.getCodeParam[Region](1)
      val in: Value[InputBuffer] = mb.getCodeParam[InputBuffer](2)
      val skip = _buildSkip(mb, r, in)
      mb.emit(skip)
    }).invokeCode(_, _)
  }

  def _buildEncoder(pt: PType, mb: EmitMethodBuilder[_], v: Value[_], out: Value[OutputBuffer]): Code[Unit]

  def _buildDecoder(pt: PType, mb: EmitMethodBuilder[_], region: Value[Region], in: Value[InputBuffer]): Code[_]

  def _buildInplaceDecoder(
    pt: PType,
    mb: EmitMethodBuilder[_],
    region: Value[Region],
    addr: Value[Long],
    in: Value[InputBuffer]
  ): Code[_] = {
    assert(!pt.isInstanceOf[PBaseStruct]) // should be overridden for structs
    val decoded = _buildDecoder(pt, mb, region, in)
    Region.storeIRIntermediate(pt)(addr, decoded)
  }

  def _buildSkip(mb: EmitMethodBuilder[_], r: Value[Region], in: Value[InputBuffer]): Code[Unit]

  def _compatible(pt: PType): Boolean = fatal("EType subclasses must override either `_compatible` or both `_encodeCompatible` and `_decodeCompatible`")

  // Can this etype encode from this ptype
  def encodeCompatible(pt: PType): Boolean = _compatible(pt)

  // Can this etype decode to this ptype
  def decodeCompatible(pt: PType): Boolean = _compatible(pt)

  final def pretty(sb: StringBuilder, indent: Int, compact: Boolean) {
    if (required)
      sb.append("+")
    _pretty(sb, indent, compact)
  }

  def asIdent: String = (if (required) "r_" else "o_") + _asIdent

  def _asIdent: String

  def _toPretty: String

  def _pretty(sb: StringBuilder, indent: Int, compact: Boolean) {
    sb.append(_toPretty)
  }

  final def decodedPType(requestedType: Type): PType = {
    val ret: PType = _decodedPType(requestedType)

    assert(decodeCompatible(ret),
      s"""Invalid requested type, cannot decode
         |encoded type  : ${ this }
         |requested type: $requestedType
         |chosen ptype  : $ret""".stripMargin)
    ret
  }

  def _decodedPType(requestedType: Type): PType

  def setRequired(required: Boolean): EType
}

trait EFundamentalType extends EType {
  def _decodeCompatible(pt: PType): Boolean = _compatible(pt)
  def _encodeCompatible(pt: PType): Boolean = _compatible(pt)
  def _buildFundamentalEncoder(pt: PType, mb: EmitMethodBuilder[_], v: Value[_], out: Value[OutputBuffer]): Code[Unit]
  def _buildFundamentalDecoder(pt: PType, mb: EmitMethodBuilder[_], region: Value[Region], in: Value[InputBuffer]): Code[_]

  final def _buildEncoder(pt: PType, mb: EmitMethodBuilder[_], v: Value[_], out: Value[OutputBuffer]): Code[Unit] =
    _buildFundamentalEncoder(pt.encodableType, mb, v, out)

  final def _buildDecoder(pt: PType, mb: EmitMethodBuilder[_], region: Value[Region], in: Value[InputBuffer]): Code[_] =
    _buildFundamentalDecoder(pt.encodableType, mb, region, in)

  final override def decodeCompatible(pt: PType): Boolean = _decodeCompatible(pt.encodableType)

  final override def encodeCompatible(pt: PType): Boolean = _encodeCompatible(pt.encodableType)
}

trait DecoderAsmFunction { def apply(r: Region, in: InputBuffer): Long }

trait EncoderAsmFunction { def apply(off: Long, out: OutputBuffer): Unit }

object EType {

  protected[encoded] def lowBitMask(n: Int): Byte = (0xFF >>> ((-n) & 0x7)).toByte
  protected[encoded] def lowBitMask(n: Code[Int]): Code[Byte] = (const(0xFF) >>> ((-n) & 0x7)).toB

  val cacheCapacity = 256
  protected val encoderCache = new util.LinkedHashMap[(EType, PType), () => EncoderAsmFunction](cacheCapacity, 0.75f, true) {
    override def removeEldestEntry(eldest: Entry[(EType, PType), () => EncoderAsmFunction]): Boolean = size() > cacheCapacity
  }
  protected var encoderCacheHits: Long = 0L
  protected var encoderCacheMisses: Long = 0L

  // The 'entry point' for building an encoder from an EType and a PType
  def buildEncoder(ctx: ExecuteContext, et: EType, pt: PType): () => EncoderAsmFunction = {
    val k = (et, pt)
    if (encoderCache.containsKey(k)) {
      encoderCacheHits += 1
      log.info(s"encoder cache hit")
      encoderCache.get(k)
    } else {
      encoderCacheMisses += 1
      log.info(s"encoder cache miss ($encoderCacheHits hits, $encoderCacheMisses misses, " +
        s"${ formatDouble(encoderCacheHits.toDouble / (encoderCacheHits + encoderCacheMisses), 3) })")
      val fb = EmitFunctionBuilder[EncoderAsmFunction](ctx, "etypeEncode",
        Array(NotGenericTypeInfo[Long], NotGenericTypeInfo[OutputBuffer]),
        NotGenericTypeInfo[Unit])
      val mb = fb.apply_method
      val f = et.buildEncoder(pt, mb.ecb)

      val addr: Code[Long] = mb.getCodeParam[Long](1)
      val out: Code[OutputBuffer] = mb.getCodeParam[OutputBuffer](2)
      val v = Region.getIRIntermediate(pt)(addr)

      mb.emit(f(v, out))
      val func = fb.result()
      encoderCache.put(k, func)
      func
    }
  }

  protected val decoderCache = new util.LinkedHashMap[(EType, Type), (PType, () => DecoderAsmFunction)](cacheCapacity, 0.75f, true) {
    override def removeEldestEntry(eldest: Entry[(EType, Type), (PType, () => DecoderAsmFunction)]): Boolean = size() > cacheCapacity
  }
  protected var decoderCacheHits: Long = 0L
  protected var decoderCacheMisses: Long = 0L

  def buildDecoder(ctx: ExecuteContext, et: EType, t: Type): (PType, () => DecoderAsmFunction) = {
    val k = (et, t)
    if (decoderCache.containsKey(k)) {
      decoderCacheHits += 1
      log.info(s"decoder cache hit")
      decoderCache.get(k)
    } else {
      decoderCacheMisses += 1
      log.info(s"decoder cache miss ($decoderCacheHits hits, $decoderCacheMisses misses, " +
        s"${ formatDouble(decoderCacheHits.toDouble / (decoderCacheHits + decoderCacheMisses), 3) }")
      val fb = EmitFunctionBuilder[DecoderAsmFunction](ctx, "etypeDecode",
        Array(NotGenericTypeInfo[Region], NotGenericTypeInfo[InputBuffer]),
        NotGenericTypeInfo[Long])
      val mb = fb.apply_method
      val pt = et.decodedPType(t)
      val f = et.buildDecoder(pt, mb.ecb)

      val region: Value[Region] = mb.getCodeParam[Region](1)
      val in: Code[InputBuffer] = mb.getCodeParam[InputBuffer](2)

      if (pt.isPrimitive) {
        val srvb = new StagedRegionValueBuilder(mb, pt, region)
        mb.emit(Code(
          srvb.start(),
          srvb.addIRIntermediate(pt)(f(region, in)),
          srvb.end()))
      } else {
        mb.emit(f(region, in))
      }

      val r = (pt, fb.result())
      decoderCache.put(k, r)
      r
    }
  }

  def defaultFromPType(pt: PType): EType = {
    pt.encodableType match {
      case t: PInt32 => EInt32(t.required)
      case t: PInt64 => EInt64(t.required)
      case t: PFloat32 => EFloat32(t.required)
      case t: PFloat64 => EFloat64(t.required)
      case t: PBoolean => EBoolean(t.required)
      case t: PBinary => EBinary(t.required)
      // FIXME(chrisvittal): turn this on when performance is adequate
      case t: PArray if t.elementType.fundamentalType.isOfType(PInt32(t.elementType.required)) &&
          HailContext.getFlag("use_packed_int_encoding") != null =>
         EPackedIntArray(t.required, t.elementType.required)
      // FIXME(chrisvittal): Turn this on when it works
      case t: PArray if t.elementType.isInstanceOf[PBaseStruct] &&
          HailContext.getFlag("use_column_encoding") != null =>
        val et = t.elementType.asInstanceOf[PBaseStruct]
        ETransposedArrayOfStructs(
          et.fields.map(f => EField(f.name, defaultFromPType(f.typ), f.index)),
          required = t.required,
          structRequired = et.required
        )
      case t: PArray => EArray(defaultFromPType(t.elementType), t.required)
      case t: PBaseStruct => EBaseStruct(t.fields.map(f =>
          EField(f.name, defaultFromPType(f.typ), f.index)), t.required)
      case t: PNDArray => ENDArrayColumnMajor(defaultFromPType(t.elementType), t.nDims, t.required)
    }
  }

  def fromTypeAndAnalysis(t: Type, r: TypeWithRequiredness): EType = t match {
    case TInt32 => EInt32(r.required)
    case TInt64 => EInt64(r.required)
    case TFloat32 => EFloat32(r.required)
    case TFloat64 => EFloat64(r.required)
    case TBoolean => EBoolean(r.required)
    case TBinary => EBinary(r.required)
    case TString => EBinary(r.required)
    case TLocus(_) =>
      EBaseStruct(Array(
        EField("contig", EBinary(true), 0),
        EField("position", EInt32(true), 1)),
        required = r.required)
    case TCall => EInt32(r.required)
    case t: TInterval =>
      val rinterval = r.asInstanceOf[RInterval]
      EBaseStruct(
        Array(
          EField("start", fromTypeAndAnalysis(t.pointType, rinterval.startType), 0),
          EField("end", fromTypeAndAnalysis(t.pointType, rinterval.endType), 1),
          EField("includesStart", EBoolean(true), 2),
          EField("includesEnd", EBoolean(true), 3)),
        required = rinterval.required)
    // FIXME(chrisvittal): turn this on when performance is adequate
    case t: TIterable if t.elementType == TInt32 &&
      HailContext.getFlag("use_packed_int_encoding") != null =>
      EPackedIntArray(r.required, coerce[RIterable](r).elementType.required)
    // FIXME(chrisvittal): Turn this on when it works
    case t: TIterable if t.elementType.isInstanceOf[TBaseStruct] &&
      HailContext.getFlag("use_column_encoding") != null =>
      val et = t.elementType.asInstanceOf[TBaseStruct]
      val rstruct = coerce[RBaseStruct](coerce[RIterable](r).elementType)
      ETransposedArrayOfStructs(
        Array.tabulate(et.size) { i =>
          val f = rstruct.fields(i)
          EField(f.name, fromTypeAndAnalysis(et.fields(i).typ, f.typ), f.index)
        }, required = r.required,
        structRequired = rstruct.required)
    case t: TIterable => EArray(fromTypeAndAnalysis(t.elementType, coerce[RIterable](r).elementType), r.required)
    case t: TBaseStruct =>
      val rstruct = coerce[RBaseStruct](r)
      EBaseStruct(Array.tabulate(t.size) { i =>
        val f = rstruct.fields(i)
        EField(f.name, fromTypeAndAnalysis(t.fields(i).typ, f.typ), f.index)
      }, required = r.required)
    case t: TNDArray =>
      val rndarray = r.asInstanceOf[RNDArray]
      ENDArrayColumnMajor(fromTypeAndAnalysis(t.elementType, rndarray.elementType), t.nDims, rndarray.required)
  }

  def eTypeParser(it: TokenIterator): EType = {
    val req = it.head match {
      case x: PunctuationToken if x.value == "+" =>
        IRParser.consumeToken(it)
        true
      case _ => false
    }

    IRParser.identifier(it) match {
      case "EBoolean" => EBoolean(req)
      case "EInt32" => EInt32(req)
      case "EInt64" => EInt64(req)
      case "EFloat32" => EFloat32(req)
      case "EFloat64" => EFloat64(req)
      case "EBinary" => EBinary(req)
      case "EPackedIntArray" =>
        IRParser.punctuation(it, "[")
        val elementsRequired = IRParser.boolean_literal(it)
        IRParser.punctuation(it, "]")
        EPackedIntArray(req, elementsRequired)
      case "EArray" =>
        IRParser.punctuation(it, "[")
        val elementType = eTypeParser(it)
        IRParser.punctuation(it, "]")
        EArray(elementType, req)
      case "EBaseStruct" =>
        IRParser.punctuation(it, "{")
        val args = IRParser.repsepUntil(it, IRParser.struct_field(eTypeParser), PunctuationToken(","), PunctuationToken("}"))
        IRParser.punctuation(it, "}")
        EBaseStruct(args.zipWithIndex.map { case ((name, t), i) => EField(name, t, i) }, req)
      case "ETransposedArrayOfStructs" =>
        IRParser.punctuation(it, "[")
        val structRequired = IRParser.boolean_literal(it)
        IRParser.punctuation(it, "]")
        IRParser.punctuation(it, "{")
        val args = IRParser.repsepUntil(it, IRParser.struct_field(eTypeParser), PunctuationToken(","), PunctuationToken("}"))
        IRParser.punctuation(it, "}")
        ETransposedArrayOfStructs(args.zipWithIndex.map { case ((name, t), i) => EField(name, t, i) }, req, structRequired)
      case "ENDArrayColumnMajor" =>
        IRParser.punctuation(it, "[")
        val elementType = eTypeParser(it)
        IRParser.punctuation(it, ",")
        val nDims = IRParser.int32_literal(it)
        IRParser.punctuation(it, "]")
        ENDArrayColumnMajor(elementType, nDims,  req)
      case x => throw new UnsupportedOperationException(s"Couldn't parse $x ${it.toIndexedSeq}")

    }
  }
}
