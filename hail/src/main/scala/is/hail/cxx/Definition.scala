package is.hail.cxx

import is.hail.cxx

trait Definition {
  def name: String
  def define: Code
}

object Variable {
  def apply(name: String, typ: Type): Variable =
    new Variable(name, typ, null)

  def apply(name: String, typ: Type, init: Code): Variable =
    new Variable(name, typ, Expression(init))

  def make_shared(prefix: String, typ: Type, constructorArgs: Code*): Variable =
    Variable(prefix, s"std::shared_ptr<$typ>", s"std::make_shared<$typ>(${ constructorArgs.mkString(", ") })")
}

class Variable(val name: String, val typ: String, init: Expression) extends Definition {
  override def toString: String = name

  def ref: Expression = Expression(name)

  def define: Code =
      if (init == null)
        s"$typ $name;"
      else s"$typ $name = $init;"

  def defineWith(value: Code): Code =
    s"$typ $name = $value;"
}

object ArrayVariable {
  def apply(prefix: String, typ: Type, len: Code): ArrayVariable =
    new ArrayVariable(prefix, typ, Expression(len))
}

class ArrayVariable(prefix: String, typ: String, length: Expression) extends Variable(prefix, typ, null) {
  override def define: Code = s"$typ $name[$length];"

  override def defineWith(value: Code): Code = s"$typ $name[$length] = $value;"
}

class Function(returnType: Type, val name: String, args: Array[Variable], body: Code, const: Boolean = false) extends Definition {

  def typ: Type = returnType

  def define: Code = s"$returnType $name(${args.map(a => s"${a.typ} ${a.name}").mkString(", ")}) ${ if (const) "const " else "" }{\n$body\n}"
}

class FunctionBuilder(val parent: ScopeBuilder, prefix: String, args: Array[Variable], returnType: Type, const: Boolean = false)
  extends ScopeBuilder with DefinitionBuilder[Function] {

  def getArg(i: Int): Variable = args(i)

  def build(): Function = new Function(returnType, prefix, args, definitions.result().mkString("\n"), const)

  def nativeError(msg: String, args: Code*): Code = s"""throw FatalError("$msg"${ args.map(a => s", ($a)").mkString });"""

}

class Class(val name: String, superClass: String, definitions: Array[Code]) extends Definition {
  def typ: Type = name

  override def toString: Type = name

  def addSuperclass(newSuper: String): Class = new Class(name, newSuper, definitions)

  def define: Code =
    s"""class $name${ if (superClass == null) "" else s" : public $superClass" } {
       |  public:
       |    ${ definitions.mkString("\n") }
       |};
     """.stripMargin
}

class ClassBuilder(val parent: ScopeBuilder, val name: String, superClass: String = null)
  extends ScopeBuilder with DefinitionBuilder[Class] {
  def build(): Class = new Class(name, superClass, definitions.result())

  def buildMethod(prefix: String, args: Array[(cxx.Type, String)], returnType: Type, const: Boolean = false): FunctionBuilder =
    new FunctionBuilder(this, prefix, args.map { case (typ, p) => new Variable(p, typ, null) }, returnType, const)
}
