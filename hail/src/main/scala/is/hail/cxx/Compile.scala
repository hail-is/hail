package is.hail.cxx

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

    val fb = FunctionBuilder("f",
      Array("Region *" -> "region", "char *" -> "v"),
      "char *")

    val v = Emit(fb, 1, body)

    fb += s"""
${v.setup}
if (${ v.m })
  abort();
return ${ v.v };
"""
    val f = fb.result()

    val sb = new StringBuilder
    sb.append(s"""
#include "hail/hail.h"
#include "hail/Utils.h"
#include "hail/Region.h"

#include <limits.h>
#include <math.h>

NAMESPACE_HAIL_MODULE_BEGIN

${ f.define }

long entrypoint(NativeStatus *st, long region, long v) {
  return (long)${ f.name }((Region *)region, (char *)v);
}

NAMESPACE_HAIL_MODULE_END
""")

    val modCode = sb.toString()

    val options = "-ggdb -O1"
    val st = new NativeStatus()
    val mod = new NativeModule(options, modCode)
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
