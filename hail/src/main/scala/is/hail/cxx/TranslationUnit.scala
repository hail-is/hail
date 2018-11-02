package is.hail.cxx

import is.hail.nativecode._
import is.hail.utils.ArrayBuilder

class TranslationUnit(preamble: String, definitions: Array[Definition]) {

  def addDefinition(d: Definition): TranslationUnit = new TranslationUnit(preamble, definitions :+ d)

  def source: String =
    new PrettyCode(
      s"""
       |$preamble
       |
       |NAMESPACE_HAIL_MODULE_BEGIN
       |
       |${definitions.map(_.define).mkString("\n\n")}
       |
       |NAMESPACE_HAIL_MODULE_END
     """.stripMargin).toString()

  def build(options: String, print: Option[java.io.PrintWriter] = None): NativeModule = {
    val st = new NativeStatus()
    val mod = new NativeModule(options, source)
    print.foreach(_.print(source))
    print.foreach(_.flush())
    mod.findOrBuild(st)
    assert(st.ok, st.toString())
    mod
  }
}

class TranslationUnitBuilder() {

  val definitions: ArrayBuilder[Definition] = new ArrayBuilder[Definition]()

  val includes: ArrayBuilder[String] = new ArrayBuilder[String]()

  def include(header: String): Unit = {
    if (header.startsWith("<") && header.endsWith(">"))
      includes += s"#include $header"
    else
      includes += s"""#include "$header""""
  }

  def +=(definition: Definition): Unit =
    definitions += definition

  def result(): TranslationUnit =
    new TranslationUnit(
      includes.result().toSet.mkString("\n"),
      definitions.result())
}