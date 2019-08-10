package is.hail.cxx

import is.hail.expr.types.physical.{PBaseStruct, PContainer}

object StagedContainerBuilder {
  def builderType(containerPType: PContainer): Type = {
    val eltType = containerPType.elementType.fundamentalType
    val params = s"${ eltType.required }, ${ eltType.byteSize }, ${ eltType.alignment }, ${ containerPType.contentsAlignment }"
    eltType match {
      case _: PBaseStruct => s"ArrayAddrBuilder<$params>"
      case t => s"ArrayLoadBuilder<${ typeToCXXType(t) }, $params>"
    }
  }
}

class StagedContainerBuilder(fb: FunctionBuilder, region: Code, containerPType: PContainer) {
  fb.translationUnitBuilder().include("hail/ArrayBuilder.h")
  private[this] val builder = fb.variable("builder", StagedContainerBuilder.builderType(containerPType))
  private[this] val i = fb.variable("i", "int", "0")

  def start(len: Code, clearMissing: Boolean = true): Code = {
    s"""
       |${ builder.defineWith(s"{ (int)$len, $region }") }
       |${ i.define }
       |${ if (clearMissing) s"$builder.clear_missing_bits();" else "" }
     """.stripMargin
  }

  def add(x: Code): Code = s"$builder.set_element($i, $x);"

  def setMissing(): Code = s"$builder.set_missing($i);"

  def advance(): Code = s"++$i;"

  def idx: Code = i.toString

  def eltOffset: Code = s"$builder.element_address($i)"

  def end(): Code = s"$builder.offset()"
}
