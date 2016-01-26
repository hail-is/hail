package org.broadinstitute.hail.expr

import org.broadinstitute.hail.Utils._
import scala.util.parsing.combinator.JavaTokenParsers

object Parser extends JavaTokenParsers {
  def parse[T](symTab: Map[String, (Int, Type)], a: Array[Any], code: String): () => T = {
    println(s"code = $code")
    val t: AST = parseAll(expr, code) match {
      case Success(result, _) => result.asInstanceOf[AST]
      case NoSuccess(msg, _) => fatal(msg)
    }
    println(s"t = $t")

    t.typecheck(symTab)
    val f: () => Any = t.eval((symTab, a))
    () => f().asInstanceOf[T]
  }

  def parseExportArgs(symTab: Map[String, (Int, Type)],
    a: Array[Any],
    code: String): (Option[String], Array[() => Any]) = {
    val (header, ts) = parseAll(export_args, code) match {
      case Success(result, _) => result.asInstanceOf[(Option[String], Array[AST])]
      case NoSuccess(msg, _) => fatal(msg)
    }

    ts.foreach(_.typecheck(symTab))
    val fs = ts.map { t =>
      t.eval((symTab, a))
    }
    (header, fs)
  }

  def expr: Parser[AST] = if_expr | or_expr

  def if_expr: Parser[AST] =
    ("if" ~> "(" ~> expr <~ ")") ~ expr ~ ("else" ~> expr) ^^ { case cond ~ thenTree ~ elseTree =>
      If(cond, thenTree, elseTree)
    }

  def or_expr: Parser[AST] =
    and_expr ~ rep(("||" | "|") ~ and_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => BinaryOp(acc, op, rhs) }
    }

  def and_expr: Parser[AST] =
    lt_expr ~ rep(("&&" | "&") ~ lt_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => BinaryOp(acc, op, rhs) }
    }

  def lt_expr: Parser[AST] =
    eq_expr ~ rep(("<=" | ">=" | "<" | ">") ~ eq_expr) ^^ { case lhs ~ lst =>
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
    tilde_expr ~ rep(("*" | "/" | "%") ~ tilde_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => BinaryOp(acc, op, rhs) }
    }

  def tilde_expr: Parser[AST] =
    apply_expr ~ rep("~" ~ apply_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => BinaryOp(acc, op, rhs) }
    }

  def export_args: Parser[(Option[String], Array[AST])] =
    // FIXME | not backtracking properly.  Why?
    args ^^ { a => (None, a) } |||
      named_args ^^ { a =>
        (Some(a.map(_._1).mkString("\t")), a.map(_._2))
      }

  def named_args: Parser[Array[(String, AST)]] =
    named_arg ~ rep("," ~ named_arg) ^^ { case arg ~ lst =>
      (arg :: lst.map { case _ ~ arg => arg }).toArray
    }

  def named_arg: Parser[(String, AST)] =
    ident ~ "=" ~ expr ^^ { case id ~ _ ~ expr => (id, expr) }

  def args: Parser[Array[AST]] =
    expr ~ rep("," ~> expr) ^^ { case arg ~ lst => (arg :: lst).toArray }

  def apply_expr: Parser[AST] =
    dot_expr ~ opt(args) ^^ {
      case f ~ Some(args) => Apply(f, args)
      case f ~ None => f
    }

  def dot_expr: Parser[AST] =
    primary_expr ~ rep(("." ~ ident) | ("(" ~ args ~ ")")) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) {
        case (acc, "." ~ sym) => Select(acc, sym)
        case (acc, "(" ~ (args: Array[AST]) ~ ")") => Apply(acc, args)
      }
    }

  def primary_expr: Parser[AST] =
    """-?\d+\.\d+[dD]?""".r ^^ (r => Const(r.toDouble, TDouble)) |
      """-?\d+(\.\d*)?[eE][+-]?\d+[dD]?""".r ^^ (r => Const(r.toDouble, TDouble)) |
      // FIXME L suffix
      wholeNumber ^^ (r => Const(r.toInt, TInt)) |
      stringLiteral ^^ { r =>
        assert(r.head == '"' && r.last == '"')
        Const(r.tail.init, TString)
      } |
      "true" ^^ (_ => Const(true, TBoolean)) |
      "false" ^^ (_ => Const(false, TBoolean)) |
      guard(not("if" | "else")) ~> ident ^^ (r => SymRef(r)) |
      "{" ~> expr <~ "}" |
      "(" ~> expr <~ ")"
}
