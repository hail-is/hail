package is.hail.cxx

trait Definition {
  def name: String
  def typ: String
  def define: Statement
}

class Variable(prefix: String, val typ: String, init: Expression) extends Definition {
  val name: String = genSym(prefix)

  override def toString: String = name

  def define: Statement = new Statement {
    override def toString: String =
      if (init == null)
        s"$typ $name"
      else s"$typ $name = $init"
  }
}

class Function(returnType: String, prefix: String, args: Array[Variable], body: Block) extends Definition {
  val name: String = genSym(prefix)

  def typ: String = returnType

  def define: Statement = Statement(s"$returnType $name(${args.map(a => s"${a.typ} ${a.name}").mkString(", ")}) $body")
}

object FunctionBuilder {

  def apply(prefix: String, args: Array[(String, String)], returnType: String): FunctionBuilder =
    new FunctionBuilder(
      prefix,
      args.map { case (typ, p) => new Variable(p, typ, null) },
      returnType)

  def apply(prefix: String, argTypes: Array[String], returnType: String): FunctionBuilder =
    apply(prefix, argTypes.map(_ -> genSym("arg")), returnType)
}

class FunctionBuilder(prefix: String, args: Array[Variable], returnType: String) {

  private[this] val blockBuilder = new BlockBuilder

  def getArg(i: Int): Variable = args(i)

  def +=(statement: Statement): Unit =
    blockBuilder += statement

  def ++=(block: Block): Unit =
    blockBuilder ++= block

  def result(): Function = new Function(returnType, prefix, args, blockBuilder.result())
}