package is.hail.cxx

import is.hail.cxx
import is.hail.expr.types.physical.PContainer

class StagedContainerBuilder(st: Code, region: Code, containerPType: PContainer) {
  private[this] val aoff = cxx.Variable("aoff", "char *")
  private[this] val eltOff = cxx.Variable("eltOff", "char *")
  private[this] val i = cxx.Variable("i", "int", "0")
  private[this] val eltRequired = containerPType.elementType.required

  def start(len: Code, clearMissing: Boolean = true): Code = {
    val __len = cxx.Variable("len", "int", len)
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
      s"""NATIVE_ERROR($st, 1010, "Required array element cannot be missing.");"""
    else
      s"set_bit($aoff + 4, $i);"
  }

  def advance(): Code = s"++$i;"

  def idx: Code = i.toString

  def eltOffset: Code = s"$eltOff + $i * ${ containerPType.elementType.byteSize }"

  def end(): Code = aoff.toString
}
