package org.broadinstitute.hail.expr

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.utils.StringEscapeUtils._

import scala.util.parsing.combinator.JavaTokenParsers
import scala.util.parsing.input.Position

case class SimplePosition(line: Int, column: Int, lineContents: String) extends Position

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

  def parseExportArgs(code: String, ec: EvalContext): (Option[Array[String]], Array[() => Option[Any]]) = {
    val (header, ts) = parseAll(export_args, code) match {
      case Success(result, _) => result.asInstanceOf[(Option[Array[String]], Array[AST])]
      case NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
    }

    ts.foreach(_.typecheck(ec))
    val fs = ts.map { t =>
      val f = t.eval(ec)
      () => Option(f())
    }

    (header, fs)
  }

  def parseNamedArgs(code: String, ec: EvalContext): (Array[String], Array[() => Option[Any]]) = {
    val args = parseAll(named_args, code) match {
      case Success(result, _) => result.asInstanceOf[Array[(String, AST)]]
      case NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
    }
    val names = args.map(_._1)
    val fns = args.map(_._2)
    fns.foreach(_.typecheck(ec))
    (names, fns.map { t =>
      val f = t.eval(ec)
      () => Option(f())
    })
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

  def export_args: Parser[(Option[Array[String]], Array[AST])] =
  // FIXME | not backtracking properly.  Why?
    args ^^ { a => (None, a) } |||
      named_args ^^ { a =>
        (Some(a.map(_._1)), a.map(_._2))
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

  def tsvIdentifier: Parser[String] = backtickLiteral | """[^\s\p{Cntrl}=,]+""".r

  def identifier = backtickLiteral | ident

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
      | withPos("[") ~ expr ~ "]"
      | withPos("[") ~ opt(expr) ~ ":" ~ opt(expr) ~ "]") ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { (acc, t) => (t: @unchecked) match {
        case (dot: Positioned[_]) ~ sym => Select(dot.pos, acc, sym)
        case (dot: Positioned[_]) ~ (sym: String) ~ "(" ~ (args: Array[AST]) ~ ")" =>
          ApplyMethod(dot.pos, acc, sym, args)
        case (lbracket: Positioned[_]) ~ (idx: AST) ~ "]" => IndexOp(lbracket.pos, acc, idx)
        case (lbracket: Positioned[_]) ~ (idx1: Option[_]) ~ ":" ~ (idx2: Option[_]) ~ "]" =>
          SliceArray(lbracket.pos, acc, idx1.map(_.asInstanceOf[AST]), idx2.map(_.asInstanceOf[AST]))
      }
      }
    }

  def primary_expr: Parser[AST] =
    withPos("""-?\d*\.\d+[dD]?""".r) ^^ (r => Const(r.pos, r.x.toDouble, TDouble)) |
      withPos("""-?\d+(\.\d*)?[eE][+-]?\d+[dD]?""".r) ^^ (r => Const(r.pos, r.x.toDouble, TDouble)) |
      // FIXME L suffix
      withPos(wholeNumber) ^^ (r => Const(r.pos, r.x.toInt, TInt)) |
      withPos(stringLiteral) ^^ { r => Const(r.pos, r.x, TString) } |
      withPos("NA" ~> ":" ~> type_expr) ^^ (r => Const(r.pos, null, r.x)) |
      withPos(arrayDeclaration) ^^ (r => ArrayConstructor(r.pos, r.x)) |
      withPos(structDeclaration) ^^ (r => StructConstructor(r.pos, r.x.map(_._1), r.x.map(_._2))) |
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

  def structField: Parser[(String, AST)] = (identifier ~ ":" ~ expr) ^^ { case id ~ _ ~ ast => (id, ast) }

  def backtickLiteral: Parser[String] =
    """`([^\\`]|\\[\\bfnrt'"`]|\\u[a-fA-F0-9]{4})*`""".r ^^
      (s => unescapeString(s.substring(1, s.length - 1))) |
      withPos("`.*`".r) ^^ { r =>
        val toSearch = r.x.substring(1, r.x.length - 1)
        val matches = """\\[^\\bfnrt`]""".r.findFirstMatchIn(toSearch)
        assert(matches.isDefined)
        val m = matches.get
        val newPos = SimplePosition(r.pos.line, r.pos.column + m.start + 1, r.pos.longString)
        ParserUtils.error(newPos,
          s"""invalid character in backtick identifier: `${
            escapeString(toSearch.charAt(m.start).toString, backticked = true)
          }'""")
      }

  override def stringLiteral: Parser[String] =
    """"([^"\\]|\\[\\bfnrt'"`]|\\u[a-fA-F0-9]{4})*"""".r ^^
      (s => unescapeString(s.substring(1, s.length - 1))) |
      withPos("""".*"""".r) ^^ { r =>
        val toSearch = r.x.substring(1, r.x.length - 1)
        val matches = """\\[^\\"bfnrt'"`]""".r.findFirstMatchIn(toSearch)
        assert(matches.isDefined)
        val m = matches.get
        val newPos = SimplePosition(r.pos.line, r.pos.column + m.start + 1, r.pos.longString)
        ParserUtils.error(newPos,
          s"""invalid character in string literal: `${
            escapeString(toSearch.charAt(m.start).toString)
          }'""")
      }

  def decorator: Parser[(String, String)] =
    ("@" ~> (identifier <~ "=")) ~ stringLiteral ^^ { case name ~ desc =>
      //    ("@" ~> (identifier <~ "=")) ~ stringLiteral("\"" ~> "[^\"]".r <~ "\"") ^^ { case name ~ desc =>
      (name, desc)
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
      ("Dict" ~ "[") ~> type_expr <~ "]" ^^ { elementType => TDict(elementType) } |
      ("Struct" ~ "{") ~> type_fields <~ "}" ^^ { fields =>
        TStruct(fields)
      }
}
