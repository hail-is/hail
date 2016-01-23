package org.broadinstitute.hail.expr

import scala.util.parsing.combinator.JavaTokenParsers

class Parser extends JavaTokenParsers {
  def expr: Parser[AST] = or_expr

  def or_expr: Parser[AST] =
    and_expr ~ rep(("|" | "||") ~ and_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => BinaryOp(acc, op, rhs) }
    }

  def and_expr: Parser[AST] =
    lt_expr ~ rep(("&" | "&&") ~ lt_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => BinaryOp(acc, op, rhs) }
    }

  def lt_expr: Parser[AST] =
    eq_expr ~ rep(("<" | "<=" | ">" | ">=") ~ eq_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => Comparison(acc, op, rhs) }
    }

  def eq_expr: Parser[AST] =
    add_expr ~ rep(("==" | "!=") ~ add_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => Comparison(acc, op, rhs) }
    }

  def add_expr: Parser[AST] =
    mul_expr ~ rep(("+" | "-") ~ mul_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => BinaryOp(acc, op, rhs) }
    }

  def mul_expr: Parser[AST] =
    dot_expr ~ rep(("*" | "/" | "%") ~ dot_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => BinaryOp(acc, op, rhs) }
    }

  def dot_expr: Parser[AST] =
    primary_expr ~ opt("." ~ ident) ^^ { case lhs ~ None => lhs
    case lhs ~ Some(_ ~ rhs) => Select(lhs, rhs)
    }

  def primary_expr: Parser[AST] = wholeNumber ^^ (r => Const(r.toInt, TInt)) |
    // FIXME L suffix
    floatingPointNumber ^^ (r => Const(r.toDouble, TDouble)) |
    stringLiteral ^^ (r => Const(r, TString)) |
    ident ^^ (r => SymRef(r)) |
    "(" ~> expr <~ ")"
}
