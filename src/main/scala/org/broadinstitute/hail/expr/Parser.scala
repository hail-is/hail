package org.broadinstitute.hail.expr

import org.broadinstitute.hail.Utils._

import scala.util.parsing.combinator.JavaTokenParsers
import scala.util.parsing.input.Position

object ParserUtils {
  def error(pos: Position, msg: String): Nothing = {
    val lineContents = pos.longString.split("\n").head
    val prefix = s"<input>:${pos.line}:"
    fatal(
      s"""$msg
         |$prefix$lineContents
         |${" " * prefix.length}${
        lineContents.take(pos.column - 1).map { c => if (c == '\t') c else ' ' }
      }^""".stripMargin)
  }
}

object Parser extends JavaTokenParsers {
  def parse(code: String, ec: EvalContext): (BaseType, () => Option[Any]) = {
    // println(s"code = $code")
    val t: AST = parseAll(expr, code) match {
      case Success(result, _) => result
      case NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
    }

    t.typecheck(ec)

    val f: () => Any = t.eval(ec)
    (t.`type`, () => Option(f()))
  }

  def parse[T](code: String, ec: EvalContext, expected: Type): () => Option[T] = {
    val (t, f) = parse(code, ec)
    if (t != expected)
      fatal(s"expression has wrong type: expected `$expected', got $t")

    () => f().map(_.asInstanceOf[T])
  }

  def parseType(code: String): Type = {
    // println(s"code = $code")
    parseAll(type_expr, code) match {
      case Success(result, _) => result
      case NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
    }
  }

  def parseAnnotationTypes(code: String): Map[String, Type] = {
    // println(s"code = $code")
    if (code.matches("""\s*"""))
      Map.empty[String, Type]
    else
      parseAll(type_fields, code) match {
        case Success(result, _) => result.map(f => (f.name, f.`type`)).toMap
        case NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
      }
  }

  def withPos[T](p: => Parser[T]): Parser[Positioned[T]] =
    positioned[Positioned[T]](p ^^ { x => Positioned(x) })

  def parseExportArgs(code: String, ec: EvalContext): (Option[String], Array[() => Option[Any]]) = {
    val (header, ts) = parseAll(export_args, code) match {
      case Success(result, _) => result.asInstanceOf[(Option[String], Array[AST])]
      case NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
    }

    ts.foreach(_.typecheck(ec))
    val fs = ts.map { t =>
      val f = t.eval(ec)
      () => Option(f())
    }

    (header, fs)
  }

  def parseAnnotationArgs(code: String, ec: EvalContext): (Array[(List[String], Type, () => Option[Any])]) = {
    val arr = parseAll(annotationExpressions, code) match {
      case Success(result, _) => result.asInstanceOf[Array[(List[String], AST)]]
      case NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
    }

    def checkType(l: List[String], t: BaseType): Type = {
      t match {
        case tws: Type => tws
        case _ => fatal(
          s"""Annotations must be stored as types with schema.
              |  Got invalid type `$t' from the result of `${l.mkString(".")}'""".stripMargin)
      }
    }

    arr.map {
      case (path, ast) =>
        ast.typecheck(ec)
        val t = checkType(path, ast.`type`)
        val f = ast.eval(ec)
        (path, t, () => Option(f()))
    }
  }

  def parseAnnotationRoot(code: String, root: String): List[String] = {
    val path = parseAll(annotationIdentifier, code) match {
      case Success(result, _) => result.asInstanceOf[List[String]]
      case NoSuccess(msg, _) => fatal(msg)
    }

    if (path.isEmpty)
      fatal(s"expected an annotation path starting in `$root', but got an empty path")
    else if (path.head != root)
      fatal(s"expected an annotation path starting in `$root', but got a path starting in '${path.head}'")
    else
      path.tail
  }

  def parseExprs(code: String, ec: EvalContext): (Array[(BaseType, () => Option[Any])]) = {

    if (code.matches("""\s*"""))
      Array.empty[(BaseType, () => Option[Any])]
    else {
      val asts = parseAll(args, code) match {
        case Success(result, _) => result.asInstanceOf[Array[AST]]
        case NoSuccess(msg, _) => fatal(msg)
      }

      asts.map { ast =>
        ast.typecheck(ec)
        val f = ast.eval(ec)
        (ast.`type`, () => Option(f()))
      }
    }
  }

  def expr: Parser[AST] = ident ~ withPos("=>") ~ expr ^^ { case param ~ arrow ~ body =>
    Lambda(arrow.pos, param, body)
  } |
    if_expr |
    let_expr |
    or_expr

  def if_expr: Parser[AST] =
    withPos("if") ~ ("(" ~> expr <~ ")") ~ expr ~ ("else" ~> expr) ^^ { case ifx ~ cond ~ thenTree ~ elseTree =>
      If(ifx.pos, cond, thenTree, elseTree)
    }

  def let_expr: Parser[AST] =
    withPos("let") ~ rep1sep((identifier <~ "=") ~ expr, "and") ~ ("in" ~> expr) ^^ { case let ~ bindings ~ body =>
      Let(let.pos, bindings.iterator.map { case id ~ v => (id, v) }.toArray, body)
    }

  def or_expr: Parser[AST] =
    and_expr ~ rep(withPos("||" | "|") ~ and_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => BinaryOp(op.pos, acc, op.x, rhs) }
    }

  def and_expr: Parser[AST] =
    lt_expr ~ rep(withPos("&&" | "&") ~ lt_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => BinaryOp(op.pos, acc, op.x, rhs) }
    }

  def lt_expr: Parser[AST] =
    eq_expr ~ rep(withPos("<=" | ">=" | "<" | ">") ~ eq_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => Comparison(op.pos, acc, op.x, rhs) }
    }

  def eq_expr: Parser[AST] =
    add_expr ~ rep(withPos("==" | "!=") ~ add_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => Comparison(op.pos, acc, op.x, rhs) }
    }

  def add_expr: Parser[AST] =
    mul_expr ~ rep(withPos("+" | "-") ~ mul_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => BinaryOp(op.pos, acc, op.x, rhs) }
    }

  def mul_expr: Parser[AST] =
    tilde_expr ~ rep(withPos("*" | "/" | "%") ~ tilde_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => BinaryOp(op.pos, acc, op.x, rhs) }
    }

  def tilde_expr: Parser[AST] =
    unary_expr ~ rep(withPos("~") ~ unary_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => BinaryOp(op.pos, acc, op.x, rhs) }
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
    tsvIdentifier ~ "=" ~ expr ^^ { case id ~ _ ~ expr => (id, expr) }

  def annotationExpressions: Parser[Array[(List[String], AST)]] =
    rep1sep(annotationExpression, ",") ^^ {
      _.toArray
    }

  def annotationExpression: Parser[(List[String], AST)] = annotationIdentifier ~ "=" ~ expr ^^ {
    case id ~ eq ~ expr => (id, expr)
  }

  def annotationIdentifier: Parser[List[String]] =
    rep1sep(identifier, ".") ^^ {
      _.toList
    }

  def annotationIdentifierArray: Parser[Array[List[String]]] =
    rep1sep(annotationIdentifier, ",") ^^ {
      _.toArray
    }

  def tsvIdentifier: Parser[String] = tickIdentifier | """[^\s\p{Cntrl}=,]+""".r

  def tickIdentifier: Parser[String] = """`[^`]+`""".r ^^ { i => i.substring(1, i.length - 1) }

  def identifier = tickIdentifier | ident

  def args: Parser[Array[AST]] =
    repsep(expr, ",") ^^ {
      _.toArray
    }

  def unary_expr: Parser[AST] =
    rep(withPos("-" | "!")) ~ dot_expr ^^ { case lst ~ rhs =>
      lst.foldRight(rhs) { case (op, acc) =>
        UnaryOp(op.pos, op.x, acc)
      }
    }

  def dot_expr: Parser[AST] =
    primary_expr ~ rep((withPos(".") ~ identifier ~ "(" ~ args ~ ")")
      | (withPos(".") ~ identifier)
      | withPos("[") ~ expr ~ ":" ~ expr ~ "]"
      | withPos("[") ~ expr ~ "]") ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { (acc, t) => (t: @unchecked) match {
        case (dot: Positioned[_]) ~ sym => Select(dot.pos, acc, sym)
        case (dot: Positioned[_]) ~ (sym: String) ~ "(" ~ (args: Array[AST]) ~ ")" => ApplyMethod(dot.pos, acc, sym, args)
        case (lbracket: Positioned[_]) ~ (idx1: AST) ~ ":" ~ (idx2: AST) ~ "]" => SliceArray(lbracket.pos, acc, idx1, idx2)
        case (lbracket: Positioned[_]) ~ (idx: AST) ~ "]" => IndexArray(lbracket.pos, acc, idx)
      }
      }
    }

  // """"([^"\p{Cntrl}\\]|\\[\\'"bfnrt])*"""".r
  def evalStringLiteral(lit: String): String = {
    assert(lit.head == '"' && lit.last == '"')
    val r = """\\[\\'"bfnrt]""".r
    // replacement does backslash expansion
    r.replaceAllIn(lit.tail.init, _.matched)
  }

  def primary_expr: Parser[AST] =
    withPos("""-?\d*\.\d+[dD]?""".r) ^^ (r => Const(r.pos, r.x.toDouble, TDouble)) |
      withPos("""-?\d+(\.\d*)?[eE][+-]?\d+[dD]?""".r) ^^ (r => Const(r.pos, r.x.toDouble, TDouble)) |
      // FIXME L suffix
      withPos(wholeNumber) ^^ (r => Const(r.pos, r.x.toInt, TInt)) |
      withPos(""""([^"\p{Cntrl}\\]|\\[\\'"bfnrt])*"""".r) ^^ { r =>
        Const(r.pos, evalStringLiteral(r.x), TString)
      } |
      withPos("NA" ~> ":" ~> type_expr) ^^ (r => Const(r.pos, null, r.x)) |
      withPos(arrayDeclaration) ^^ (r => ArrayDeclaration(r.pos, r.x)) |
      withPos(structDeclaration) ^^ (r => StructDeclaration(r.pos, r.x.map(_._1), r.x.map(_._2))) |
      withPos("true") ^^ (r => Const(r.pos, true, TBoolean)) |
      withPos("false") ^^ (r => Const(r.pos, false, TBoolean)) |
      (guard(not("if" | "else")) ~> withPos(identifier)) ~ withPos("(") ~ (args <~ ")") ^^ {
        case id ~ lparen ~ args =>
          Apply(lparen.pos, id.x, args)
      } |
      guard(not("if" | "else")) ~> withPos(identifier) ^^ (r => SymRef(r.pos, r.x)) |
      "{" ~> expr <~ "}" |
      "(" ~> expr <~ ")"

  def annotationSignature: Parser[TStruct] =
    type_fields ^^ { fields => TStruct(fields) }

  def arrayDeclaration: Parser[Array[AST]] = "[" ~> repsep(expr, ",") <~ "]" ^^ (_.toArray)

  def structDeclaration: Parser[Array[(String, AST)]] = "{" ~> repsep(structField, ",") <~ "}" ^^ (_.toArray)

  def structField: Parser[(String, AST)] = (stringLiteral ~ ":" ~ expr) ^^ {case id ~ _ ~ ast => (id, ast)}

  def decorator: Parser[(String, String)] =
    ("@" ~> (identifier <~ "=")) ~ stringLiteral ^^ { case name ~ desc =>
      (unescapeString(name), {
        val unescaped = unescapeString(desc)
        unescaped.substring(1, unescaped.length - 1)
      })
    }

  def type_field: Parser[(String, Type, Map[String, String])] =
    (identifier <~ ":") ~ type_expr ~ rep(decorator) ^^ { case name ~ t ~ decorators =>
      (name, t, decorators.toMap)
    }

  def type_fields: Parser[Array[Field]] = repsep(type_field, ",") ^^ {
    _.zipWithIndex.map { case ((id, t, attrs), index) => Field(id, t, index, attrs) }
      .toArray
  }

  def type_expr: Parser[Type] =
    "Empty" ^^ { _ => TStruct.empty } |
      "Boolean" ^^ { _ => TBoolean } |
      "Char" ^^ { _ => TChar } |
      "Int" ^^ { _ => TInt } |
      "Long" ^^ { _ => TLong } |
      "Float" ^^ { _ => TFloat } |
      "Double" ^^ { _ => TDouble } |
      "String" ^^ { _ => TString } |
      "Sample" ^^ { _ => TSample } |
      "AltAllele" ^^ { _ => TAltAllele } |
      "Variant" ^^ { _ => TVariant } |
      "Genotype" ^^ { _ => TGenotype } |
      "String" ^^ { _ => TString } |
      ("Array" ~ "[") ~> type_expr <~ "]" ^^ { elementType => TArray(elementType) } |
      ("Set" ~ "[") ~> type_expr <~ "]" ^^ { elementType => TSet(elementType) } |
      ("Struct" ~ "{") ~> type_fields <~ "}" ^^ { fields =>
        TStruct(fields)
      }
}
