package is.hail.cxx

import java.io.FileOutputStream

import is.hail.expr.ir
import is.hail.expr.types.physical._
import is.hail.nativecode.NativeStatus
import is.hail.utils.fatal

import scala.reflect.classTag

object Compile {
  var i = 0
  def apply(
    arg0: String, arg0Type: PType,
    body: ir.IR, optimize: Boolean): (Long, Long) => Long = {
    assert(ir.TypeToIRIntermediateClassTag(arg0Type.virtualType) == classTag[Long])
    assert(arg0Type.isInstanceOf[PBaseStruct])
    val returnType = body.pType

    assert(ir.TypeToIRIntermediateClassTag(returnType.virtualType) == classTag[Long])
    assert(returnType.isInstanceOf[PBaseStruct])

    val tub = new TranslationUnitBuilder

    tub.include("hail/hail.h")
    tub.include("hail/Utils.h")
    tub.include("hail/Region.h")
    tub.include("hail/Upcalls.h")

    tub.include("<cstring>")

    val fb = tub.buildFunction("f",
      Array("RegionPtr" -> "region", "const char *" -> "v"),
      "char *")

    val v = Emit(fb, 1, ir.Subst(body, ir.Env(arg0 -> ir.In(0, arg0Type.virtualType))))

    fb +=
      s"""
         |UpcallEnv up;
         |${ v.setup }
         |if (${ v.m })
         |  abort();
         |return ${ v.v };
         |""".stripMargin
    val f = fb.end()

    tub += new Definition {
      def name: String = "entrypoint"

      def define: String =
        s"""
           |long entrypoint(NativeStatus *st, long region, long v) {
           |  try {
           |    return (long)${ f.name }(((ScalaRegion *)region)->region_, (char *)v);
           |  } catch (const FatalError& e) {
           |    NATIVE_ERROR(st, 1005, e.what());
           |    return -1;
           |  }
           |}
         """.stripMargin
    }

    val tu = tub.end()
    val mod = tu.build(if (optimize) "-ggdb -O1" else "-ggdb -O0")
    i += 1

    val st = new NativeStatus()
    mod.findOrBuild(st)
    assert(st.ok, st.toString())
    val nativef = mod.findLongFuncL2(st, "entrypoint")
    assert(st.ok, st.toString())

    // mod will be cleaned up when f is closed
    mod.close()
    st.close()

    { (v1: Long, v2: Long) =>
      val st2 = new NativeStatus()
      val res = nativef(st2, v1, v2)
      if (st2.fail)
        fatal(st2.toString())
      res
    }
  }
}
