package is.hail.cxx

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import is.hail.HailContext
import is.hail.annotations.{Region, RegionValueBuilder}
import is.hail.expr.ir
import is.hail.expr.ir.BindingEnv
import is.hail.expr.types.physical._
import is.hail.io.CodecSpec
import is.hail.nativecode.{NativeModule, NativeStatus, ObjectArray}
import is.hail.utils.{SerializableHadoopConfiguration, fatal}
import is.hail.utils.richUtils.RichHadoopConfiguration

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
    tub.include("hail/Hadoop.h")
    tub.include("hail/SparkUtils.h")
    tub.include("hail/ObjectArray.h")

    tub.include("<cstring>")

    val fb = tub.buildFunction(tub.genSym("f"),
      (("SparkFunctionContext", "ctx") +: ("HadoopConfig", "hadoop_config") +: args.map { case (_, typ) => typeToCXXType(typ) -> "v" }).toArray,
      typeToCXXType(body.pType))

    val emitEnv = args.zipWithIndex
      .foldLeft(ir.Env[ir.IR]()){ case (env, ((arg, argType), i)) =>
        env.bind(arg -> ir.In(i, argType.virtualType)) }
    val (v, mods, emitLiterals) = Emit(fb, ir.Streamify(ir.Subst(body, BindingEnv(emitEnv))))

    val (literals, litvars) = emitLiterals.unzip
    val litType = PTuple(literals.map { case (t, _) => t })
    val f = { spec: CodecSpec =>

      val baos = new ByteArrayOutputStream()
      val enc = spec.buildEncoder(litType)(baos)
      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)
        rvb.start(litType)
        rvb.startTuple()
        literals.foreach { case (t, a) => rvb.addAnnotation(t.virtualType, a) }
        rvb.endTuple()
        enc.writeRegionValue(region, rvb.end())
      }
      enc.flush()
      enc.close()
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
         |return ${ v.v };
         |""".stripMargin

    (fb.end(), mods, (litType, f))
  }

  def makeEntryPoint(tub: TranslationUnitBuilder, literals: EncodedLiterals, fname: String, argTypes: PType*): Literals = {
    val nArgs = argTypes.size
    val rawArgs = if (nArgs == 0) "" else
      Array.tabulate(nArgs)(i => s"long v$i").mkString(", ", ", ", "")
    val castArgs = if (nArgs == 0) "" else
      Array.tabulate(nArgs)(i => s"(${ typeToNonConstCXXType(argTypes(i)) }) v$i").mkString(", ", ", ", "")

    val (lType, encoded) = literals
    val litEnc = defaultSpec.buildNativeDecoderClass(lType, lType, tub).name

    tub.include("hail/Upcalls.h")

    tub += new Definition {
      def name: String = "entrypoint"

      def define: String =
        s"""
           |long entrypoint(NativeStatus *st, long obj, long jregion $rawArgs) {
           |  try {
           |    UpcallEnv up;
           |    RegionPtr region = ((ScalaRegion *)jregion)->region_;
           |    jobject sparkUtils = ((ObjectArray *) obj)->at(0);
           |    jobject jlit_in = ((ObjectArray *) obj)->at(1);
           |    $litEnc lit_in { std::make_shared<InputStream>(up, jlit_in) };
           |    const char * lit_ptr = lit_in.decode_row(region.get());
           |
           |    jobject jhadoop_config = ((ObjectArray *) obj)->at(2);
           |    HadoopConfig hadoop_config(up, jhadoop_config);
           |
           |    return (long)$fname(SparkFunctionContext(region, sparkUtils, lit_ptr), hadoop_config $castArgs);
           |  } catch (const FatalError& e) {
           |    NATIVE_ERROR(st, 1005, e.what());
           |    return -1;
           |  }
           |}
         """.stripMargin
    }
    encoded(defaultSpec)
  }

  def compile(body: ir.IR, optimize: Boolean, args: Array[(String, PType)]): (NativeModule, SparkUtils, Literals) = {
    val tub = new TranslationUnitBuilder
    val (f, mods, literals) = makeNonmissingFunction(tub, body, args: _*)
    val encLiterals = makeEntryPoint(tub, literals, f.name, args.map(_._2): _*)

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

    val hadoopConf = new SerializableHadoopConfiguration(HailContext.get.sc.hadoopConfiguration)

    { (region: Long) =>
      val st2 = new NativeStatus()
      val javaArgs = new ObjectArray(sparkUtils, new ByteArrayInputStream(literals),
        new RichHadoopConfiguration(hadoopConf.value)).get()

      val res = nativef(st2, javaArgs, region)
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

    { (region: Long, v2: Long) =>
      val st2 = new NativeStatus()
      val res = nativef(st2, new ObjectArray(sparkUtils, new ByteArrayInputStream(literals)).get(), region, v2)
      if (st2.fail)
        fatal(st2.toString())
      res
    }
  }
}
