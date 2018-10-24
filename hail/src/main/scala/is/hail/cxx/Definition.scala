package is.hail.cxx

import is.hail.utils.ArrayBuilder

trait Definition {
  def name: String
  def typ: String
  def define: Code
}

object Variable {
  def apply(prefix: String, typ: Type): Variable =
    new Variable(prefix, typ, null)

  def apply(prefix: String, typ: Type, init: Code): Variable =
    new Variable(prefix, typ, Expression(init))
}

class Variable(prefix: String, val typ: String, init: Expression) extends Definition {
  val name: String = genSym(prefix)

  override def toString: String = name

  def toExpr: Expression = Expression(name)

  def define: Code =
      if (init == null)
        s"$typ $name;"
      else s"$typ $name = $init;"
}

class ArrayVariable(prefix: String, typ: String, length: Expression) extends Variable(prefix, typ, null) {
  override def define: Code = s"$typ $name[$length];"
}

class Function(returnType: Type, val name: String, args: Array[Variable], body: Code) extends Definition {

  def typ: Type = returnType

  def define: Code = s"$returnType $name(${args.map(a => s"${a.typ} ${a.name}").mkString(", ")}) {\n$body\n}"
}

object FunctionBuilder {

  def apply(prefix: String, args: Array[(Type, String)], returnType: Type): FunctionBuilder =
    new FunctionBuilder(
      prefix,
      args.map { case (typ, p) => new Variable(p, typ, null) },
      returnType)

  def apply(prefix: String, argTypes: Array[Type], returnType: Type): FunctionBuilder =
    apply(prefix, argTypes.map(_ -> genSym("arg")), returnType)
}

class FunctionBuilder(prefix: String, args: Array[Variable], returnType: Type) {

  val statements: ArrayBuilder[Code] = new ArrayBuilder[Code]()

  def +=(statement: Code) =
    statements += statement

  def getArg(i: Int): Variable = args(i)

  def result(): Function = new Function(returnType, prefix, args, statements.result().mkString("\n"))

}
