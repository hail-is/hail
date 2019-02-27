package is.hail.cxx

import is.hail.expr.types.physical.{PBaseStruct, PContainer, PType}

class StagedContainerBuilder(fb: FunctionBuilder, region: Code, containerPType: PContainer) {
  fb.translationUnitBuilder().include("hail/ArrayBuilder.h")

  val eltType: PType = containerPType.elementType.fundamentalType

  private[this] val params = s"${ eltType.required }, ${ eltType.byteSize }, ${ eltType.alignment }, ${ containerPType.contentsAlignment }"
  private[this] val builderType: Type = eltType match {
    case _: PBaseStruct => s"ArrayAddrBuilder<$params>"
    case t => s"ArrayLoadBuilder<${typeToCXXType(t)}, $params>"
  }
  private[this] val builder = fb.variable("builder", builderType)
  private[this] val i = fb.variable("i", "int", "0")

  def start(len: Code, clearMissing: Boolean = true): Code = {
    s"""
       |${ builder.defineWith(s"{ (int)$len, $region }") }
       |${ i.define }
       |${ if (clearMissing) s"$builder.clear_missing_bits();" }
     """.stripMargin
  }

  def add(x: Code): Code = s"$builder.set_element($i, $x);"

  def setMissing(): Code = s"$builder.set_missing($i);"

  def advance(): Code = s"++$i;"

  def idx: Code = i.toString

  def eltOffset: Code = s"$builder.element_address($i)"

  def end(): Code = s"$builder.offset()"
}
