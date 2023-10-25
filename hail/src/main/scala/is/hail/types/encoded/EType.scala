package is.hail.types.encoded
import is.hail.annotations.Region
import is.hail.asm4s.{coerce => _, _}
import is.hail.backend.ExecuteContext
import is.hail.expr.ir.{EmitClassBuilder, EmitCodeBuilder, EmitFunctionBuilder, EmitMethodBuilder, IRParser, ParamType, PunctuationToken, TokenIterator}
import is.hail.io._
import is.hail.types._
import is.hail.types.physical._
import is.hail.types.physical.stypes.{SType, SValue}
import is.hail.types.virtual._
import is.hail.utils._
import org.json4s.CustomSerializer
import org.json4s.JsonAST.JString

import java.util
import java.util.Map.Entry


class ETypeSerializer extends CustomSerializer[EType](format => ( {
  case JString(s) => IRParser.parse[EType](s, EType.eTypeParser)
}, {
  case t: EType => JString(t.parsableString())
}))


abstract class EType extends BaseType with Serializable with Requiredness {
  type StagedEncoder = (EmitCodeBuilder, SValue, Value[OutputBuffer]) => Unit
  type StagedDecoder = (EmitCodeBuilder, Value[Region], Value[InputBuffer]) => SValue
  type StagedInplaceDecoder = (EmitCodeBuilder, Value[Region], Value[Long], Value[InputBuffer]) => Unit

  final def buildEncoder(ctx: ExecuteContext, t: PType): (OutputBuffer, ExecuteContext) => Encoder = {
    val f = EType.buildEncoder(ctx, this, t)
    (out: OutputBuffer, ctx: ExecuteContext) => new CompiledEncoder(out, ctx, f)
  }

  final def buildDecoder(ctx: ExecuteContext, requestedType: Type): (PType, (InputBuffer, HailClassLoader) => Decoder) = {
    val (rt, f) = EType.buildDecoderToRegionValue(ctx, this, requestedType)
    val makeDec = (in: InputBuffer, theHailClassLoader: HailClassLoader) =>
      new CompiledDecoder(in, rt, theHailClassLoader, f)
    (rt, makeDec)
  }

  final def buildStructDecoder(ctx: ExecuteContext, requestedType: TStruct): (PStruct, (InputBuffer, HailClassLoader) => Decoder) = {
    val (pType: PStruct, makeDec) = buildDecoder(ctx, requestedType)
    pType -> makeDec
  }

  final def buildEncoder(st: SType, kb: EmitClassBuilder[_]): StagedEncoder = {
    val mb = buildEncoderMethod(st, kb);
    { (cb: EmitCodeBuilder, sv: SValue, ob: Value[OutputBuffer]) => cb.invokeVoid(mb, sv, ob) }
  }

  final def buildEncoderMethod(st: SType, kb: EmitClassBuilder[_]): EmitMethodBuilder[_] = {
    kb.getOrGenEmitMethod(s"ENCODE_${ st.asIdent }_TO_${ asIdent }",
      (st, this, "ENCODE"),
      FastSeq[ParamType](st.paramType, classInfo[OutputBuffer]),
      UnitInfo) { mb =>

      mb.voidWithBuilder { cb =>
        val arg = mb.getSCodeParam(1)
        val out = mb.getCodeParam[OutputBuffer](2)
        _buildEncoder(cb, arg, out)
      }
    }
  }

  final def buildDecoder(t: Type, kb: EmitClassBuilder[_]): StagedDecoder = {
    val mb = buildDecoderMethod(t: Type, kb);
    { (cb: EmitCodeBuilder, r: Value[Region], ib: Value[InputBuffer]) =>
      cb.invokeSCode(mb, r, ib)
    }
  }

  final def buildDecoderMethod[T](t: Type, kb: EmitClassBuilder[_]): EmitMethodBuilder[_] = {
    val st = decodedSType(t)
    kb.getOrGenEmitMethod(s"DECODE_${ asIdent }_TO_${ st.asIdent }",
      (t, this, "DECODE"),
      FastSeq[ParamType](typeInfo[Region], classInfo[InputBuffer]),
      st.paramType) { mb =>

      mb.emitSCode { cb =>
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
    { (cb: EmitCodeBuilder, r: Value[Region], addr: Value[Long], ib: Value[InputBuffer]) =>
      cb.invokeVoid(mb, r, addr, ib)
    }
  }

  final def buildInplaceDecoderMethod(pt: PType, kb: EmitClassBuilder[_]): EmitMethodBuilder[_] = {
    kb.getOrGenEmitMethod(s"INPLACE_DECODE_${ asIdent }_TO_${ pt.asIdent }",
      (pt, this, "INPLACE_DECODE"),
      FastSeq[ParamType](typeInfo[Region], typeInfo[Long], classInfo[InputBuffer]),
      UnitInfo)({ mb =>

      mb.voidWithBuilder { cb =>
        val region: Value[Region] = mb.getCodeParam[Region](1)
        val addr: Value[Long] = mb.getCodeParam[Long](2)
        val in: Value[InputBuffer] = mb.getCodeParam[InputBuffer](3)
        _buildInplaceDecoder(cb, pt, region, addr, in)
      }
    })
  }

  final def buildSkip(kb: EmitClassBuilder[_]): (EmitCodeBuilder, Value[Region], Value[InputBuffer]) => Unit = {
    val mb = kb.getOrGenEmitMethod(s"SKIP_${ asIdent }",
      (this, "SKIP"),
      FastSeq[ParamType](classInfo[Region], classInfo[InputBuffer]),
      UnitInfo)({ mb =>
      mb.voidWithBuilder { cb =>
        val r: Value[Region] = mb.getCodeParam[Region](1)
        val in: Value[InputBuffer] = mb.getCodeParam[InputBuffer](2)
        _buildSkip(cb, r, in)
      }
    })

    { (cb, r, in) => cb.invokeVoid(mb, r, in) }
  }

  def _buildEncoder(cb: EmitCodeBuilder, v: SValue, out: Value[OutputBuffer]): Unit

  def _buildDecoder(cb: EmitCodeBuilder, t: Type, region: Value[Region], in: Value[InputBuffer]): SValue

  def _buildInplaceDecoder(
    cb: EmitCodeBuilder,
    pt: PType,
    region: Value[Region],
    addr: Value[Long],
    in: Value[InputBuffer]
  ): Unit = {
    assert(!pt.isInstanceOf[PBaseStruct]) // should be overridden for structs
    val decoded = _buildDecoder(cb, pt.virtualType, region, in)
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
    decodedSType(requestedType).storageType().setRequired(required)
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
  protected val encoderCache = new util.LinkedHashMap[(EType, PType), (ExecuteContext) => EncoderAsmFunction](cacheCapacity, 0.75f, true) {
    override def removeEldestEntry(eldest: Entry[(EType, PType), (ExecuteContext) => EncoderAsmFunction]): Boolean = size() > cacheCapacity
  }
  protected var encoderCacheHits: Long = 0L
  protected var encoderCacheMisses: Long = 0L

  // The 'entry point' for building an encoder from an EType and a PType
  def buildEncoder(ctx: ExecuteContext, et: EType, pt: PType): (ExecuteContext) => EncoderAsmFunction = {
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
        val addr: Value[Long] = mb.getCodeParam[Long](1)
        val out: Value[OutputBuffer] = mb.getCodeParam[OutputBuffer](2)
        val pc = pt.loadCheapSCode(cb, addr)
        val f = et.buildEncoder(pc.st, mb.ecb)
        f(cb, pc, out)
      }
      val compiledFunc = {
        val result = fb.resultWithIndex()
        (ctx: ExecuteContext) => result(ctx.theHailClassLoader, ctx.fs, ctx.taskContext, ctx.r)
      }
      val func = fb.resultWithIndex()
      encoderCache.put(k, compiledFunc)
      compiledFunc
    }
  }

  protected val decoderCache = new util.LinkedHashMap[(EType, Type), (PType, (HailClassLoader) => DecoderAsmFunction)](cacheCapacity, 0.75f, true) {
    override def removeEldestEntry(eldest: Entry[(EType, Type), (PType, (HailClassLoader) => DecoderAsmFunction)]): Boolean = size() > cacheCapacity
  }
  protected var decoderCacheHits: Long = 0L
  protected var decoderCacheMisses: Long = 0L

  def buildDecoderToRegionValue(ctx: ExecuteContext, et: EType, t: Type): (PType, (HailClassLoader) => DecoderAsmFunction) = {
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
      val in: Value[InputBuffer] = mb.getCodeParam[InputBuffer](2)

      mb.emitWithBuilder[Long] { cb =>
        val pc = f(cb, region, in)
        pt.store(cb, region, pc, false)
      }

      val compiledFunc = {
        val result = fb.resultWithIndex()
        (hcl: HailClassLoader) => result(hcl, ctx.fs, ctx.taskContext, ctx.r)
      }
      val r = (pt, compiledFunc)
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
    case TString => EBinary(r.required)
    case TLocus(_) =>
      EBaseStruct(Array(
        EField("contig", EBinary(true), 0),
        EField("position", EInt32(true), 1)),
        required = r.required)
    case TCall => EInt32(r.required)
    case TRNGState => ERNGState(r.required, None)
    case t: TInterval =>
      val rinterval = r.asInstanceOf[RInterval]
      EBaseStruct(
        Array(
          EField("start", fromTypeAndAnalysis(t.pointType, rinterval.startType), 0),
          EField("end", fromTypeAndAnalysis(t.pointType, rinterval.endType), 1),
          EField("includesStart", EBoolean(true), 2),
          EField("includesEnd", EBoolean(true), 3)),
        required = rinterval.required)
    case t: TIterable => EArray(fromTypeAndAnalysis(t.elementType, tcoerce[RIterable](r).elementType), r.required)
    case t: TBaseStruct =>
      val rstruct = tcoerce[RBaseStruct](r)
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

  def fromTypeAllOptional(t: Type): EType = t match {
    case TInt32 => EInt32(false)
    case TInt64 => EInt64(false)
    case TFloat32 => EFloat32(false)
    case TFloat64 => EFloat64(false)
    case TBoolean => EBoolean(false)
    case TBinary => EBinary(false)
    case TString => EBinary(false)
    case TLocus(_) =>
      EBaseStruct(Array(
        EField("contig", EBinary(false), 0),
        EField("position", EInt32(false), 1)),
        required = false)
    case TCall => EInt32(false)
    case t: TInterval =>
      EBaseStruct(
        Array(
          EField("start", fromTypeAllOptional(t.pointType), 0),
          EField("end", fromTypeAllOptional(t.pointType), 1),
          EField("includesStart", EBoolean(false), 2),
          EField("includesEnd", EBoolean(false), 3)),
        required = false)
    case t: TIterable => EArray(fromTypeAllOptional(t.elementType), false)
    case t: TBaseStruct =>
      EBaseStruct(Array.tabulate(t.size) { i =>
        val f = t.fields(i)
        if (f.index != i)
          throw new AssertionError(s"${t} [$i]")
        EField(f.name, fromTypeAllOptional(t.fields(i).typ), f.index)
      }, required = false)
    case t: TNDArray =>
      ENDArrayColumnMajor(fromTypeAllOptional(t.elementType), t.nDims, false)
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
