package is.hail.cxx

import java.io.PrintStream

import is.hail.expr.types.physical.PType
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
    assert(st.ok, st.toString() + "\n" + source)
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

  def variable(prefix: String, typ: String, init: Code): Variable = {
    Variable(genSym(prefix), typ, init)
  }

  def variable(prefix: String, typ: String): Variable = {
    Variable(genSym(prefix), typ)
  }

  def arrayVariable(prefix: String, typ: String, len: Code): ArrayVariable = {
    ArrayVariable(genSym(prefix), typ, len)
  }

  def make_shared(prefix: String, typ: String, args: Code*): Variable = {
    Variable.make_shared(genSym(prefix), typ, args: _*)
  }

  def translationUnitBuilder(): TranslationUnitBuilder = {
    this match {
      case tub: TranslationUnitBuilder => tub
      case _ => parent.translationUnitBuilder()
    }
  }

  def ordering(lp: PType, rp: PType): String = {
    val tub = translationUnitBuilder()
    tub.orderings.ordering(tub, lp, rp)
  }

  def genSym(name: String): String = translationUnitBuilder().genSym(name)

  def genErr: Long = translationUnitBuilder().genErr
}

class TranslationUnitBuilder() extends ScopeBuilder {
  val parent: ScopeBuilder = null

  val orderings: Orderings = new Orderings

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
      args.map { case (typ, p) => variable(p, typ) },
      returnType)
  }

  def buildClass(name: String, superClass: String = null): ClassBuilder =
    new ClassBuilder(this, name, superClass)

  private var symCounter: Long = 0

  override def genSym(name: String): String = {
    symCounter += 1
    s"$name$symCounter"
  }

  override def genErr: Long = symCounter

  def end(): TranslationUnit = {

    new TranslationUnit(
      includes.result().distinct.mkString("\n"),
      definitions.result())
  }
}