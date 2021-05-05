package is.hail.types.encoded
import java.util
import java.util.Map.Entry

import is.hail.HailContext
import is.hail.annotations.Region
import is.hail.asm4s.{coerce => _, _}
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder, EmitFunctionBuilder, EmitMethodBuilder, ExecuteContext, IRParser, ParamType, PunctuationToken, TokenIterator}
import is.hail.io._
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.physical.stypes.{SCode, SType}
import is.hail.types.virtual._
import is.hail.utils._
import org.json4s.CustomSerializer
import org.json4s.JsonAST.JString


class ETypeSerializer extends CustomSerializer[EType](format => ( {
  case JString(s) => IRParser.parse[EType](s, EType.eTypeParser)
}, {
  case t: EType => JString(t.parsableString())
}))


abstract class EType extends BaseType with Serializable with Requiredness {
  type StagedEncoder = (EmitCodeBuilder, SCode, Code[OutputBuffer]) => Unit
  type StagedDecoder = (EmitCodeBuilder, Code[Region], Code[InputBuffer]) => PCode
  type StagedInplaceDecoder = (EmitCodeBuilder, Code[Region], Code[Long], Code[InputBuffer]) => Unit

  final def buildEncoder(ctx: ExecuteContext, t: PType): (OutputBuffer) => Encoder = {
    val f = EType.buildEncoder(ctx, this, t)
    out: OutputBuffer => new CompiledEncoder(out, f)
  }

  final def buildDecoder(ctx: ExecuteContext, requestedType: Type): (PType, (InputBuffer) => Decoder) = {
    val (rt, f) = EType.buildDecoderToRegionValue(ctx, this, requestedType)
    (rt, (in: InputBuffer) => new CompiledDecoder(in, f))
  }

  final def buildStructDecoder(ctx: ExecuteContext, requestedType: TStruct): (PStruct, (InputBuffer) => Decoder) = {
    val (pType: PStruct, makeDec) = buildDecoder(ctx, requestedType)
    pType -> makeDec
  }

  final def buildEncoder(st: SType, kb: EmitClassBuilder[_]): StagedEncoder = {
    val mb = buildEncoderMethod(st, kb);
    { (cb: EmitCodeBuilder, sc: SCode, ob: Code[OutputBuffer]) => cb.invokeVoid(mb, sc.asPCode, ob) }
  }

  final def buildEncoderMethod(st: SType, kb: EmitClassBuilder[_]): EmitMethodBuilder[_] = {
    kb.getOrGenEmitMethod(s"ENCODE_${ st.asIdent }_TO_${ asIdent }",
      (st, this, "ENCODE"),
      FastIndexedSeq[ParamType](st.paramType, classInfo[OutputBuffer]),
      UnitInfo) { mb =>

      mb.voidWithBuilder { cb =>
        val arg = mb.getPCodeParam(1)
          .memoize(cb, "encoder_method_arg")
        val out = mb.getCodeParam[OutputBuffer](2)
        _buildEncoder(cb, arg, out)
      }
    }
  }

  final def buildDecoder(t: Type, kb: EmitClassBuilder[_]): StagedDecoder = {
    val mb = buildDecoderMethod(t: Type, kb);
    { (cb: EmitCodeBuilder, r: Code[Region], ib: Code[InputBuffer]) =>
      cb.invokePCode(mb, r, ib)
    }
  }

  final def buildDecoderMethod[T](t: Type, kb: EmitClassBuilder[_]): EmitMethodBuilder[_] = {
    val st = decodedSType(t)
    kb.getOrGenEmitMethod(s"DECODE_${ asIdent }_TO_${ st.asIdent }",
      (t, this, "DECODE"),
      FastIndexedSeq[ParamType](typeInfo[Region], classInfo[InputBuffer]),
      st.paramType) { mb =>

      mb.emitPCode { cb =>
        val region: Value[Region] = mb.getCodeParam[Region](1)
        val in: Value[InputBuffer] = mb.getCodeParam[InputBuffer](2)
        val sc = _buildDecoder(cb, t, region, in)
        if (sc.st != st)
          throw new RuntimeException(s"decoder type mismatch:\n  inferred: $st\n  returned: ${ sc.st }")
        sc
      }
    }
  }

  final def buildInplaceDecoder(pt: PType, kb: EmitClassBuilder[_]): StagedInplaceDecoder = {
    val mb = buildInplaceDecoderMethod(pt, kb);
    { (cb: EmitCodeBuilder, r: Code[Region], addr: Code[Long], ib: Code[InputBuffer]) =>
      cb.invokeVoid(mb, r, addr, ib)
    }
  }

  final def buildInplaceDecoderMethod(pt: PType, kb: EmitClassBuilder[_]): EmitMethodBuilder[_] = {
    kb.getOrGenEmitMethod(s"INPLACE_DECODE_${ asIdent }_TO_${ pt.asIdent }",
      (pt, this, "INPLACE_DECODE"),
      FastIndexedSeq[ParamType](typeInfo[Region], typeInfo[Long], classInfo[InputBuffer]),
      UnitInfo)({ mb =>

      mb.voidWithBuilder { cb =>
        val region: Value[Region] = mb.getCodeParam[Region](1)
        val addr: Value[Long] = mb.getCodeParam[Long](2)
        val in: Value[InputBuffer] = mb.getCodeParam[InputBuffer](3)
        _buildInplaceDecoder(cb, pt, region, addr, in)
      }
    })
  }

  final def buildSkip(mb: EmitMethodBuilder[_]): (Code[Region], Code[InputBuffer]) => Code[Unit] = {
    mb.getOrGenEmitMethod(s"SKIP_${ asIdent }",
      (this, "SKIP"),
      FastIndexedSeq[ParamType](classInfo[Region], classInfo[InputBuffer]),
      UnitInfo)({ mb =>
      mb.voidWithBuilder { cb =>
        val r: Value[Region] = mb.getCodeParam[Region](1)
        val in: Value[InputBuffer] = mb.getCodeParam[InputBuffer](2)
        _buildSkip(cb, r, in)
      }
    }).invokeCode(_, _)
  }

  def _buildEncoder(cb: EmitCodeBuilder, v: PValue, out: Value[OutputBuffer]): Unit

  def _buildDecoder(cb: EmitCodeBuilder, t: Type, region: Value[Region], in: Value[InputBuffer]): PCode

  def _buildInplaceDecoder(
    cb: EmitCodeBuilder,
    pt: PType,
    region: Value[Region],
    addr: Value[Long],
    in: Value[InputBuffer]
  ): Unit = {
    assert(!pt.isInstanceOf[PBaseStruct]) // should be overridden for structs
    val decoded = _buildDecoder(cb, pt.virtualType, region, in).memoize(cb, "Asd")
    pt.storeAtAddress(cb, addr, region, decoded, false)
  }

  def _buildSkip(cb: EmitCodeBuilder, r: Value[Region], in: Value[InputBuffer]): Unit

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

  final def decodedSType(requestedType: Type): SType = {
    _decodedSType(requestedType)
  }

  final def decodedPType(requestedType: Type): PType = {
    decodedSType(requestedType).canonicalPType()
  }

  def _decodedSType(requestedType: Type): SType

  def setRequired(required: Boolean): EType
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

      mb.voidWithBuilder { cb =>
        val addr: Code[Long] = mb.getCodeParam[Long](1)
        val out: Code[OutputBuffer] = mb.getCodeParam[OutputBuffer](2)
        val pc = pt.loadCheapPCode(cb, addr)
        val f = et.buildEncoder(pc.st, mb.ecb)
        f(cb, pc, out)
      }
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

  def buildDecoderToRegionValue(ctx: ExecuteContext, et: EType, t: Type): (PType, () => DecoderAsmFunction) = {
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
      val f = et.buildDecoder(t, mb.ecb)

      val region: Value[Region] = mb.getCodeParam[Region](1)
      val in: Code[InputBuffer] = mb.getCodeParam[InputBuffer](2)

      mb.emitWithBuilder[Long] { cb =>
        val pc = f(cb, region, in)
        pt.store(cb, region, pc, false)
      }

      val r = (pt, fb.result())
      decoderCache.put(k, r)
      r
    }
  }

  def defaultFromPType(pt: PType): EType = {
    val r = VirtualTypeWithReq(pt)
    fromTypeAndAnalysis(r.t, r.r)
  }

  def fromTypeAndAnalysis(t: Type, r: TypeWithRequiredness): EType = t match {
    case TInt32 => EInt32(r.required)
    case TInt64 => EInt64(r.required)
    case TFloat32 => EFloat32(r.required)
    case TFloat64 => EFloat64(r.required)
    case TBoolean => EBoolean(r.required)
    case TBinary => EBinary(r.required)
    case _: TShuffle => EShuffle(r.required)
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
    case t: TIterable => EArray(fromTypeAndAnalysis(t.elementType, coerce[RIterable](r).elementType), r.required)
    case t: TBaseStruct =>
      val rstruct = coerce[RBaseStruct](r)
      assert(t.size == rstruct.size, s"different number of fields: ${t} ${r}")
      EBaseStruct(Array.tabulate(t.size) { i =>
        val f = rstruct.fields(i)
        if (f.index != i)
          throw new AssertionError(s"${t} [$i]")
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
