package is.hail.cxx

import is.hail.expr.ir
import is.hail.expr.types.physical._
import is.hail.nativecode.{NativeModule, NativeStatus, ObjectArray}
import is.hail.utils.fatal

import scala.reflect.classTag

object Compile {

  def makeNonmissingFunction(tub: TranslationUnitBuilder, body: ir.IR, args: (String, PType)*): (Function, Array[(String, NativeModule)]) = {
    tub.include("hail/hail.h")
    tub.include("hail/Utils.h")
    tub.include("hail/Region.h")
    tub.include("hail/Upcalls.h")
    tub.include("hail/SparkUtils.h")
    tub.include("hail/ObjectArray.h")

    tub.include("<cstring>")

    val fb = tub.buildFunction(tub.genSym("f"),
      (("SparkFunctionContext", "ctx") +: args.map { case (_, typ) =>
        typeToCXXType(typ) -> "v" }).toArray,
      typeToCXXType(body.pType))

    val emitEnv = args.zipWithIndex
      .foldLeft(ir.Env[ir.IR]()){ case (env, ((arg, argType), i)) =>
        env.bind(arg -> ir.In(i, argType.virtualType)) }
    val (v, mods) = Emit(fb, 1, ir.Subst(body, emitEnv))

    fb +=
      s"""
         |${ v.setup }
         |if (${ v.m })
         |  abort();
         |return ${ v.v };
         |""".stripMargin
    (fb.end(), mods)
  }


  def apply(
    arg0: String, arg0Type: PType,
    body: ir.IR, optimize: Boolean): (Long, Long) => Long = {
    assert(ir.TypeToIRIntermediateClassTag(arg0Type.virtualType) == classTag[Long])
    assert(arg0Type.isInstanceOf[PBaseStruct])
    val returnType = body.pType

    assert(ir.TypeToIRIntermediateClassTag(returnType.virtualType) == classTag[Long])
    assert(returnType.isInstanceOf[PBaseStruct])

    val tub = new TranslationUnitBuilder
    val (f, mods) = makeNonmissingFunction(tub, body, arg0 -> arg0Type)

    tub += new Definition {
      def name: String = "entrypoint"

      def define: String =
        s"""
           |long entrypoint(NativeStatus *st, long sparkUtils, long region, long v) {
           |  try {
           |    return (long)${ f.name }(SparkFunctionContext(((ScalaRegion *)region)->region_, ((ObjectArray *)sparkUtils)->at(0)), (char *)v);
           |  } catch (const FatalError& e) {
           |    NATIVE_ERROR(st, 1005, e.what());
           |    return -1;
           |  }
           |}
         """.stripMargin
    }

    val tu = tub.end()
    val mod = tu.build(if (optimize) "-ggdb -O1" else "-ggdb -O0")

    val st = new NativeStatus()
    mod.findOrBuild(st)
    assert(st.ok, st.toString())
    val nativef = mod.findLongFuncL3(st, "entrypoint")
    assert(st.ok, st.toString())

    // mod will be cleaned up when f is closed
    mod.close()
    st.close()

    val sparkUtils = new SparkUtils(mods)

    { (region: Long, v2: Long) =>
      val st2 = new NativeStatus()
      val res = nativef(st2, new ObjectArray(sparkUtils).get(), region, v2)
      if (st2.fail)
        fatal(st2.toString())
      res
    }
  }
}
