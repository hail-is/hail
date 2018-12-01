package is.hail.cxx

import java.io.PrintStream

import is.hail.nativecode._
import is.hail.utils.ArrayBuilder

class TranslationUnit(preamble: String, definitions: Array[Code]) {

  def addDefinition(d: Definition): TranslationUnit = new TranslationUnit(preamble, definitions :+ d.define)

  def source: String =
    new PrettyCode(
      s"""
         |$preamble
         |
         |NAMESPACE_HAIL_MODULE_BEGIN
         |
         |${ definitions.mkString("\n\n") }
         |
         |NAMESPACE_HAIL_MODULE_END
     """.stripMargin).toString()

  def build(options: String, s: PrintStream = null): NativeModule = {
    val src = source
    if (s != null)
      s.println(src)
    val st = new NativeStatus()
    val mod = new NativeModule(options, src)
    mod.findOrBuild(st)
    assert(st.ok, st.toString())
    mod
  }
}

trait DefinitionBuilder[T <: Definition] {
  val parent: ScopeBuilder

  protected def build(): T

  def end(): T = {
    val t = build()
    parent += t
    t
  }
}

trait ScopeBuilder {
  val parent: ScopeBuilder

  val definitions: ArrayBuilder[Code] = new ArrayBuilder[Code]

  def +=(d: Definition) {
    definitions += d.define
  }

  def +=(c: Code) {
    definitions += c
  }

  def translationUnitBuilder(): TranslationUnitBuilder = {
    this match {
      case tub: TranslationUnitBuilder => tub
      case _ => parent.translationUnitBuilder()
    }
  }
}

class TranslationUnitBuilder() extends ScopeBuilder {
  val parent: ScopeBuilder = null

  val includes: ArrayBuilder[String] = new ArrayBuilder[String]()

  def include(header: String): Unit = {
    if (header.startsWith("<") && header.endsWith(">"))
      includes += s"#include $header"
    else
      includes += s"""#include "$header""""
  }

  def buildFunction(prefix: String, args: Array[(Type, String)], returnType: Type): FunctionBuilder = {
    new FunctionBuilder(this,
      prefix,
      args.map { case (typ, p) => new Variable(p, typ, null) },
      returnType)
  }

  def buildClass(name: String, superClass: String = null): ClassBuilder =
    new ClassBuilder(this, name, superClass)

  def end(): TranslationUnit = {
    new TranslationUnit(
      includes.result().mkString("\n"),
      definitions.result())
  }
}