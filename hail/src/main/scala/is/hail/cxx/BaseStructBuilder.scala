package is.hail.cxx

import is.hail.expr.types.physical.PBaseStruct
import is.hail.utils.ArrayBuilder

class BaseStructBuilder(fb: FunctionBuilder, pStruct: PBaseStruct) {
  val ab = new ArrayBuilder[Code]
  // FIXME region is 0
  val s = Variable("s", "char *", s"${ fb.getArg(0) }->allocate(${ pStruct.alignment }, ${ pStruct.byteSize })")
  ab += s.define
  // FIXME just missing bits
  ab += s"memset($s, 0, ${ pStruct.byteSize });"

  private var i = 0

  def add(t: EmitTriplet) {
    val f = pStruct.fields(i)
    // FIXME test can be missing
    ab += s"""
${ t.setup }
if (${ t.m })
  store_bit($s, ${ pStruct.missingIdx(i) }, 1);
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
