package is.hail.cxx

import is.hail.expr.types.physical.PBaseStruct
import is.hail.utils.ArrayBuilder

class BaseStructBuilder(fb: FunctionBuilder, pStruct: PBaseStruct) {
  val ab = new ArrayBuilder[Code]
  val s = Variable("s", "char *", s"${ fb.getArg(0) }->allocate(${ pStruct.alignment }, ${ pStruct.byteSize })")
  ab += s.define
  ab += s"memset($s, 0, ${ pStruct.nMissingBytes });"

  private var i = 0

  def add(t: EmitTriplet) {
    val f = pStruct.fields(i)
    ab += s"""
${ t.setup }
if (${ t.m })
  ${
      if (pStruct.fieldRequired(i))
        "abort();"
      else
        s"store_bit($s, ${ pStruct.missingIdx(i) }, 1);"
    }
else
  ${
      storeIRIntermediate(f.typ,
        s"($s) + (${ pStruct.byteOffsets(i) })",
        t.v)
    };
"""
    i += 1
  }

  def result(): EmitTriplet = {
    assert(i == pStruct.size)

    EmitTriplet(pStruct, ab.result().mkString, "false", s.toString)
  }
}
