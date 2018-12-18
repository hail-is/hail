package is.hail.cxx

import is.hail.annotations.UnsafeUtils
import is.hail.cxx
import is.hail.expr.types.physical.PContainer

class StagedContainerBuilder(fb: FunctionBuilder, region: Code, containerPType: PContainer) {
  private[this] val aoff = fb.variable("aoff", "char *")
  private[this] val eltOff = fb.variable("eltOff", "char *")
  private[this] val i = fb.variable("i", "int", "0")
  private[this] val eltRequired = containerPType.elementType.required

  def start(len: Code, clearMissing: Boolean = true): Code = {
    val __len = fb.variable("len", "int", len)
    s"""${ __len.define }
       |${ start(__len, clearMissing) }
     """.stripMargin
  }

  def start(len: Variable, clearMissing: Boolean): Code = {
    val nMissingBytes = containerPType.cxxNMissingBytes(len.toString)
    val elementsOffset = containerPType.cxxElementsOffset(len.toString)
    val allocate = s"${ region }->allocate(${ containerPType.contentsAlignment }, ${ containerPType.cxxContentsByteSize(len.toString) })"

    s"""${ aoff.defineWith(allocate) }
       |${ eltOff.defineWith(s"$aoff + $elementsOffset") }
       |${ i.define }
       |store_int($aoff, ${ len });
       |${ if (!eltRequired && clearMissing) s"memset($aoff + 4, 0, $nMissingBytes);" else "" }
     """.stripMargin

  }

  def add(x: Code): Code = {
    s"${
      storeIRIntermediate(containerPType.elementType,
        eltOffset,
        x)
    };"
  }

  def setMissing(): Code = {
    if (eltRequired)
      fb.nativeError("Required array element cannot be missing.")
    else
      s"set_bit($aoff + 4, $i);"
  }

  def advance(): Code = s"++$i;"

  def idx: Code = i.toString

  def eltOffset: Code = s"$eltOff + $i * ${ UnsafeUtils.arrayElementSize(containerPType.elementType) }"

  def end(): Code = aoff.toString
}
