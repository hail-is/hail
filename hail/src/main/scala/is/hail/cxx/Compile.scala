package is.hail.cxx

import java.io.PrintWriter

import is.hail.expr.ir
import is.hail.expr.types.physical._
import is.hail.nativecode.{NativeLongFuncL2, NativeModule, NativeStatus}

import scala.reflect.classTag

object Compile {
  def apply(
    arg0: String, arg0Type: PType,
    body: ir.IR): NativeLongFuncL2 = {
    assert(ir.TypeToIRIntermediateClassTag(arg0Type.virtualType) == classTag[Long])
    assert(arg0Type.isInstanceOf[PBaseStruct])
    val returnType = body.pType

    assert(ir.TypeToIRIntermediateClassTag(returnType.virtualType) == classTag[Long])
    assert(returnType.isInstanceOf[PBaseStruct])

    val tub = new TranslationUnitBuilder

    tub.include("hail/hail.h")
    tub.include("hail/Utils.h")
    tub.include("hail/Region.h")

    tub.include("<limits.h>")
    tub.include("<math.h>")

    val fb = tub.buildFunction("f",
      Array("NativeStatus *" -> "st", "ScalaRegion *" -> "region", "const char *" -> "v"),
      "char *")

    val v = Emit(fb, 2, body)

    fb +=
      s"""
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
           |  return (long)${ f.name }(st, (ScalaRegion *)region, (char *)v);
           |}
         """.stripMargin
    }

    val tu = tub.end()
    val mod = tu.build("-ggdb -O1")

    val st = new NativeStatus()
    mod.findOrBuild(st)
    assert(st.ok, st.toString())
    val nativef = mod.findLongFuncL2(st, "entrypoint")
    assert(st.ok, st.toString())

    // mod will be cleaned up when f is closed
    mod.close()
    st.close()

    nativef
  }
}
