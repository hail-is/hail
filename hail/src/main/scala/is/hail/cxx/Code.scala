package is.hail.cxx

import is.hail.utils.ArrayBuilder

object Code {
  def apply(s: String): Code = new Code {
    override def toString: String = s
  }
}

abstract class Code {
  def toString: String
}

object Statement {
  def apply(s: String): Statement = new Statement {
    override def toString: String = s
  }
}

abstract class Statement extends Code

object Expression {
  def apply(s: String): Expression = new Expression {
    override def toString: String = s
  }

}

abstract class Expression extends Code

object Block {
  def apply(statements: Statement*): Block = new Block(statements.toArray)
}

class Block(val statements: Array[Statement]) extends Statement {

  override def toString: String = s"{\n${statements.mkString("\n")}\n}"

}

class BlockBuilder() {
  private[this] val statements: ArrayBuilder[Statement] = new ArrayBuilder[Statement]

  def +=(s: Statement): Unit =
    statements += s

  def +=(s: String): Unit =
    statements += Statement(s)

  def ++=(block: Block): Unit =
    block.statements.foreach(statements += _)

  def ++=(bb: BlockBuilder): Unit =
    ++=(bb.result())

  def result(): Block = new Block(statements.result())
}