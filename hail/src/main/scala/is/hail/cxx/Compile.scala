package is.hail.cxx

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import is.hail.HailContext
import is.hail.annotations.{Region, RegionValueBuilder}
import is.hail.expr.ir
import is.hail.expr.ir.BindingEnv
import is.hail.expr.types.physical._
import is.hail.expr.types.virtual.TVoid
import is.hail.io.CodecSpec
import is.hail.nativecode.{NativeModule, NativeStatus, ObjectArray}
import is.hail.utils.fatal

import scala.reflect.classTag

object Compile {

  type Literals = Array[Byte]
  type EncodedLiterals = (PTuple, CodecSpec => Literals)

  val defaultSpec: CodecSpec = CodecSpec.defaultUncompressed

  def makeNonmissingFunction(tub: TranslationUnitBuilder, body: ir.IR, args: (String, PType)*): (Function, Array[(String, (Literals, NativeModule))], EncodedLiterals) = {
    tub.include("hail/hail.h")
    tub.include("hail/Utils.h")
    tub.include("hail/Region.h")
    tub.include("hail/Upcalls.h")
    tub.include("hail/FS.h")
    tub.include("hail/SparkUtils.h")
    tub.include("hail/ObjectArray.h")

    tub.include("<cstring>")

    val fb = tub.buildFunction(tub.genSym("f"),
      (("SparkFunctionContext", "ctx") +: args.map { case (_, typ) => typeToCXXType(typ) -> "v" }).toArray,
      typeToCXXType(body.pType))

    val emitEnv = args.zipWithIndex
      .foldLeft(ir.Env[ir.IR]()){ case (env, ((arg, argType), i)) =>
        env.bind(arg -> ir.In(i, argType.virtualType)) }
    val (v, mods, emitLiterals) = Emit(fb, ir.Streamify(ir.Subst(body, BindingEnv(emitEnv))))

    val (literals, litvars) = emitLiterals.unzip
    val litType = PTuple(literals.map { case (t, _) => t }: _*)
    val f = { spec: CodecSpec =>

      val enc = spec.makeCodecSpec2(litType)
      val baos = new ByteArrayOutputStream()
      val encoder = enc.buildEncoder(litType)(baos)
      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)
        rvb.start(litType)
        rvb.startTuple()
        literals.foreach { case (t, a) => rvb.addAnnotation(t.virtualType, a) }
        rvb.endTuple()
        encoder.writeRegionValue(region, rvb.end())
      }
      encoder.flush()
      encoder.close()
      baos.toByteArray
    }

    val ctxLit = s"${ fb.getArg(0).name }.literals_"
    val litSetup = Array.tabulate(literals.length) { i =>
      litvars(i).defineWith(litType.cxxLoadField(ctxLit, i))
    }.mkString("\n")

    fb +=
      s"""
         |$litSetup
         |${ v.setup }
         |if (${ v.m })
         |  abort();
         |${ if (body.typ != TVoid) s"return ${ v.v };" else "" }
         |""".stripMargin

    (fb.end(), mods, (litType, f))
  }

  def makeEntryPoint(tub: TranslationUnitBuilder, literals: EncodedLiterals, fname: String, isVoid: Boolean,
    argTypes: PType*): Literals = {
    val nArgs = argTypes.size
    val rawArgs = if (nArgs == 0) "" else
      Array.tabulate(nArgs)(i => s"long v$i").mkString(", ", ", ", "")
    val castArgs = if (nArgs == 0) "" else
      Array.tabulate(nArgs)(i => s"(${ typeToNonConstCXXType(argTypes(i)) }) v$i").mkString(", ", ", ", "")

    val (lType, encoded) = literals
    val (lTypeRet, litEnc) = defaultSpec.makeCodecSpec2(lType).buildNativeDecoderClass(lType.virtualType, "InputStream", tub)
    assert(lTypeRet == lType)

    tub.include("hail/Upcalls.h")

    tub += new Definition {
      def name: String = "entrypoint"

      val funcCall = s"$fname(SparkFunctionContext(region, sparkUtils, fs, lit_ptr) $castArgs)"
      def define: String =
        s"""
           |long entrypoint(NativeStatus *st, long obj, long jregion $rawArgs) {
           |  try {
           |    UpcallEnv up;
           |    RegionPtr region = ((ScalaRegion *)jregion)->region_;
           |    jobject sparkUtils = ((ObjectArray *) obj)->at(0);
           |    jobject fs = ((ObjectArray *) obj)->at(1);
           |    jobject jlit_in = ((ObjectArray *) obj)->at(2);
           |    ${ litEnc.name } lit_in { std::make_shared<InputStream>(up, jlit_in) };
           |    const char * lit_ptr = lit_in.decode_row(region.get());
           |
           |    ${ if (!isVoid) s"return (long)$funcCall;" else s"$funcCall;\nreturn 0;" }
           |  } catch (const FatalError& e) {
           |    NATIVE_ERROR(st, 1005, e.what());
           |    return -1;
           |  }
           |}
         """.stripMargin
    }
    encoded(defaultSpec)
  }

  def compileComparison(op: String, codec: CodecSpec, l: PType, r: PType): Array[Byte] = {
    assert(l.isInstanceOf[PArray] || l.isInstanceOf[PBaseStruct], l)
    assert(r.isInstanceOf[PArray] || r.isInstanceOf[PBaseStruct], r)
    val tub = new TranslationUnitBuilder()
    tub.include("hail/hail.h")
    tub.include("hail/Utils.h")
    tub.include("hail/Region.h")
    tub.include("hail/Upcalls.h")
    tub.include("hail/FS.h")
    tub.include("hail/SparkUtils.h")
    tub.include("hail/ObjectArray.h")
    tub.include("hail/RegionPool.h")
    tub.include("hail/Region.h")
    tub.include("<cstring>")
    tub.include("<iostream>")
    val o = new Orderings().ordering(tub, l, r)
    val typ = if (op == "compare") "int" else "bool"
    val (lTyp, decodel) = codec.makeCodecSpec2(l).buildNativeDecoderClass(l.virtualType, "ByteArrayInputStream", tub)
    val (rTyp, decoder) = codec.makeCodecSpec2(r).buildNativeDecoderClass(r.virtualType, "ByteArrayInputStream", tub)
    assert(lTyp == l)
    assert(rTyp == r)
    tub += new Definition {
      def name = op
      def define =
        s"""extern "C" $typ $op(char *l, long lsize, char *r, long rsize) {
           |  RegionPool region_pool{};
           |  RegionPtr region = region_pool.get_region();
           |  return $o::$op($decodel(std::make_shared<ByteArrayInputStream>(l, lsize)).decode_row(region.get()),
           |                 $decoder(std::make_shared<ByteArrayInputStream>(r, rsize)).decode_row(region.get()));
           |} """.stripMargin
    }
    tub.end().build("-ggdb -O1").getBinary
  }

  def compile(body: ir.IR, optimize: Boolean, args: Array[(String, PType)]): (NativeModule, SparkUtils, Literals) = {
    val tub = new TranslationUnitBuilder
    val (f, mods, literals) = makeNonmissingFunction(tub, body, args: _*)
    val encLiterals = makeEntryPoint(tub, literals, f.name, body.typ == TVoid, args.map(_._2): _*)

    val tu = tub.end()
    val mod = tu.build(if (optimize) "-ggdb -O1" else "-ggdb -O0")

    val st = new NativeStatus()
    mod.findOrBuild(st)
    assert(st.ok, st.toString())

    (mod, new SparkUtils(mods), encLiterals)
  }

  def apply(body: ir.IR, optimize: Boolean): Long => Long = {
    val (mod, sparkUtils, literals) = compile(body, optimize, Array())

    val st = new NativeStatus()
    val nativef = mod.findLongFuncL2(st, "entrypoint")
    mod.close()
    st.close()

    val fs = HailContext.get.sFS

    { (region: Long) =>
      val st2 = new NativeStatus()
      val jObjectArgs = new ObjectArray(sparkUtils, fs, new ByteArrayInputStream(literals)).get()

      val res = nativef(st2, jObjectArgs, region)
      if (st2.fail)
        fatal(st2.toString())
      res
    }
  }

  def apply(
    arg0: String, arg0Type: PType,
    body: ir.IR, optimize: Boolean): (Long, Long) => Long = {
    assert(ir.TypeToIRIntermediateClassTag(arg0Type.virtualType) == classTag[Long])
    assert(arg0Type.isInstanceOf[PBaseStruct])
    val returnType = body.pType

    assert(ir.TypeToIRIntermediateClassTag(returnType.virtualType) == classTag[Long])
    assert(returnType.isInstanceOf[PBaseStruct])
    val (mod, sparkUtils, literals) = compile(body, optimize, Array(arg0 -> arg0Type))

    val st = new NativeStatus()
    val nativef = mod.findLongFuncL3(st, "entrypoint")
    mod.close()
    st.close()

    val fs = HailContext.get.sFS

    { (region: Long, v2: Long) =>
      val st2 = new NativeStatus()
      val jObjectArgs = new ObjectArray(sparkUtils, fs, new ByteArrayInputStream(literals)).get()
      val res = nativef(st2, jObjectArgs, region, v2)
      if (st2.fail)
        fatal(st2.toString())
      res
    }
  }
}
