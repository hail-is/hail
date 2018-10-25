package is.hail.cxx

object Expression {
  def apply(s: String): Expression = new Expression {
    override def toString: Code = s
  }
}

abstract class Expression {
  def toString: Code
}