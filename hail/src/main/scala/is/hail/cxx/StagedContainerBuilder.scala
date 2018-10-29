package is.hail.cxx

import is.hail.cxx
import is.hail.expr.types.physical.PContainer

class StagedContainerBuilder(fb: FunctionBuilder, containerPType: PContainer) {
  val a = cxx.Variable("a", "char *")
  val b = cxx.Variable("b", "char *")
  val i = cxx.Variable("i", "int", "0")

  def start(len: Code): Code = {
    val __len = cxx.Variable("len", "int", len)
    val nMissingBytes = cxx.Variable("nMissingBytes", "long", containerPType.cxxNMissingBytes(__len.toString))
    val elementsOffset = cxx.Variable("elementsOffset", "long",
      containerPType.cxxElementsOffset(__len.toString))

    s"""
${ __len.define }
${ nMissingBytes.define }
${ elementsOffset.define }
${ a.define }
${ b.define }
${ i.define }
$a = ${ fb.getArg(0) }->allocate(${ containerPType.contentsAlignment }, ${ containerPType.cxxContentsByteSize(__len.toString) });
store_int($a, ${ __len });
memset($a + 4, 0, $nMissingBytes);
$b = $a + $elementsOffset;
"""
  }

  def add(x: Code): Code = {
    s"${
      storeIRIntermediate(containerPType.elementType,
        s"$b + $i * ${ containerPType.elementType.byteSize }",
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
