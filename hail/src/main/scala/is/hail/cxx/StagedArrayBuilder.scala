package is.hail.cxx

import is.hail.cxx
import is.hail.expr.types.physical.PArray

class StagedArrayBuilder(fb: FunctionBuilder, pArray: PArray) {
  val a = cxx.Variable("a", "char *")
  val b = cxx.Variable("b", "char *")
  val i = cxx.Variable("i", "int", "0")

  def start(len: Code): Code = {
    val __len = cxx.Variable("len", "int", len)
    val nMissingBytes = cxx.Variable("nMissingBytes", "long", s"(${ __len } + 7) >> 3")
    val elementsOffset = cxx.Variable("elementsOffset", "long",
      s"round_up_alignment(4 + $nMissingBytes, ${ pArray.elementType.byteSize })")

    s"""
${ __len.define }
${ nMissingBytes.define }
${ elementsOffset.define }
${ a.define }
${ b.define }
${ i.define }
$a = ${ fb.getArg(0) }->allocate(${ pArray.alignment }, $elementsOffset + ${ __len } * ${ pArray.elementType.byteSize });
store_int($a, ${ __len });
memset($a + 4, 0, $nMissingBytes);
$b = $a + $elementsOffset;
"""
  }

  def add(x: Code): Code = {
    s"${
      storeIRIntermediate(pArray.elementType,
        s"$b + $i * ${ pArray.elementType.byteSize }",
        x)
    };"
  }

  def setMissing(): Code = {
    s"store_bit($a + 4, $i, 1);"
  }

  def advance(): Code = {
    s"++$i;"
  }

  def end(): Code = {
    a.toString
  }
}
