package org.broadinstitute.hail.expr

import org.apache.hadoop
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.utils.StringEscapeUtils._

import scala.io.Source
import scala.util.parsing.combinator.JavaTokenParsers
import scala.util.parsing.input.Position
import scala.collection.mutable

object ParserUtils {
  def error(pos: Position, msg: String): Nothing = {
    val lineContents = pos.longString.split("\n").head
    val prefix = s"<input>:${ pos.line }:"
    fatal(
      s"""$msg
         |$prefix$lineContents
         |${ " " * prefix.length }${
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

  def parseIdentifierList(code: String): Array[String] = {
    if (code.matches("""\s*"""))
      Array.empty[String]
    else
      parseAll(identifierList, code) match {
        case Success(result, _) => result
        case NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
      }
  }

  def withPos[T](p: => Parser[T]): Parser[Positioned[T]] =
    positioned[Positioned[T]](p ^^ { x => Positioned(x) })

  def parseCommaDelimitedDoubles(code: String): Array[Double] = {
    parseAll(comma_delimited_doubles, code) match {
      case Success(r, _) => r
      case NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
    }
  }
  
  def parseNamedArgs(code: String, ec: EvalContext): (Option[Array[String]], Array[Type], () => Array[String]) = {
    val result = parseAll(export_args, code) match {
      case Success(r, _) => r
      case NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
    }

    var noneNamed = true
    var allNamed = true

    val nb = mutable.ArrayBuilder.make[String]
    val tb = mutable.ArrayBuilder.make[Type]

    val computations = result.map { case (name, ast) =>
      ast.typecheck(ec)
      ast.`type` match {
        case s: TSplat =>
          val eval = ast.eval(ec)
          noneNamed = false
          s.struct.fields.map { f =>
            nb += name.map(x => s"$x.${ f.name }").getOrElse(f.name)
            tb += f.`type`
          }
          val types = s.struct.fields.map(_.`type`)
          () => eval().asInstanceOf[IndexedSeq[Any]].iterator
            .zip(types.iterator).map { case (value, t) => t.str(value) }
        case t: Type =>
          name match {
            case Some(n) =>
              noneNamed = false
              nb += n
            case None =>
              allNamed = false
          }
          tb += t
          val f = ast.eval(ec)
          () => Iterator(t.str(f()))
        case bt => fatal(s"tried to export invalid type `$bt'")
      }
    }

    if (!(noneNamed || allNamed))
      fatal(
        """export expressions require either all arguments named or none
          |  Hint: exploded structs (e.g. va.info.*) count as named arguments """.stripMargin)

    val names = nb.result()

    (someIf(names.nonEmpty, names), tb.result(), () => computations.flatMap(_ ()))
  }

  def parseExportArgs(code: String, ec: EvalContext): (Array[String], Array[Type], () => Array[String]) = {
    val (headerOption, ts, f) = parseNamedArgs(code, ec)
    val header = headerOption match {
      case Some(h) => h
      case None => fatal(
        """this module requires named export arguments
          |  e.g. `gene = va.gene, csq = va.csq' rather than `va.gene, va.csq'""".stripMargin)
    }
    (header, ts, f)
  }

  def parseAnnotationArgs(code: String, ec: EvalContext, expectedHead: Option[String]): (Array[(List[String], Type)], Array[() => Any]) = {
    val arr = parseAll(annotationExpressions, code) match {
      case Success(result, _) => result.asInstanceOf[Array[(List[String], AST)]]
      case NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
    }

    def checkType(l: List[String], t: BaseType): Type = {
      if (expectedHead.isDefined && l.head != expectedHead.get)
        fatal(
          s"""invalid annotation path `${ l.map(prettyIdentifier).mkString(".") }'
              |  Path should begin with `$expectedHead'
           """.stripMargin)

      t match {
        case t: Type => t
        case bt => fatal(
          s"""Got invalid type `$t' from the result of `${ l.mkString(".") }'""".stripMargin)
      }
    }

    val all = arr.map {
      case (path, ast) =>
        ast.typecheck(ec)
        val t = checkType(path, ast.`type`)
        val f = ast.eval(ec)
        val name = if (expectedHead.isDefined) path.tail else path
        ((name, t), () => f())
    }

    (all.map(_._1), all.map(_._2))
  }

  def parseAnnotationRoot(code: String, root: String): List[String] = {
    val path = parseAll(annotationIdentifier, code) match {
      case Success(result, _) => result.asInstanceOf[List[String]]
      case NoSuccess(msg, _) => fatal(msg)
    }

    if (path.isEmpty)
      fatal(s"expected an annotation path starting in `$root', but got an empty path")
    else if (path.head != root)
      fatal(s"expected an annotation path starting in `$root', but got a path starting in '${ path.head }'")
    else
      path.tail
  }

  def parseNamedExprs(code: String, ec: EvalContext): Array[(String, BaseType, () => Option[Any])] = {
    val parsed = parseAll(named_args, code) match {
      case Success(result, _) => result.asInstanceOf[Array[(String, AST)]]
      case NoSuccess(msg, _) => fatal(msg)
    }

    parsed.map { case (name, ast) =>
      ast.typecheck(ec)
      val f = ast.eval(ec)
      (name, ast.`type`, () => Option(f()))
    }
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

  def export_args: Parser[Array[(Option[String], AST)]] =
  // FIXME | not backtracking properly.  Why?
    rep1sep(expr ^^ { e => (None, e) } |||
      named_arg ^^ { case (name, expr) => (Some(name), expr) }, ",") ^^ (_.toArray)

  def comma_delimited_doubles: Parser[Array[Double]] =
    repsep(floatingPointNumber, ",") ^^ (_.map(_.toDouble).toArray)

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

  def identifierList: Parser[Array[String]] = rep1sep(identifier, ",") ^^ {
    _.toArray
  }

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
      | (withPos(".") ~ "*")
      | withPos("[") ~ expr ~ "]"
      | withPos("[") ~ opt(expr) ~ ":" ~ opt(expr) ~ "]") ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { (acc, t) => (t: @unchecked) match {
        case (dot: Positioned[_]) ~ "*" => Splat(dot.pos, acc)
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

  def advancePosition(pos: Position, delta: Int) = new Position {
    def line = pos.line

    def column = pos.column + delta

    def lineContents = pos.longString.split("\n").head
  }

  def quotedLiteral(delim: Char, what: String): Parser[String] =
    withPos(s"$delim([^$delim\\\\]|\\\\.)*$delim".r) ^^ { s =>
      try {
        unescapeString(s.x.substring(1, s.x.length - 1))
      } catch {
        case e: Exception =>
          val toSearch = s.x.substring(1, s.x.length - 1)
          """\\[^\\"bfnrt'"`]""".r.findFirstMatchIn(toSearch) match {
            case Some(m) =>
              // + 1 for the opening "
              ParserUtils.error(advancePosition(s.pos, m.start + 1),
                s"""invalid escape character in $what: ${ m.matched }""")

            case None =>
              // For safety.  Should never happen.
              ParserUtils.error(s.pos, "invalid $what")
          }

      }
    } | withPos(s"$delim([^$delim\\\\]|\\\\.)*\\z".r) ^^ { s =>
      ParserUtils.error(s.pos, s"unterminated $what")
    }

  def backtickLiteral: Parser[String] = quotedLiteral('`', "backtick identifier")

  override def stringLiteral: Parser[String] = quotedLiteral('"', "string literal")

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
      "Interval" ^^ { _ => TInterval } |
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
      "Locus" ^^ { _ => TLocus } |
      "Genotype" ^^ { _ => TGenotype } |
      "String" ^^ { _ => TString } |
      ("Array" ~ "[") ~> type_expr <~ "]" ^^ { elementType => TArray(elementType) } |
      ("Set" ~ "[") ~> type_expr <~ "]" ^^ { elementType => TSet(elementType) } |
      ("Dict" ~ "[") ~> type_expr <~ "]" ^^ { elementType => TDict(elementType) } |
      ("Struct" ~ "{") ~> type_fields <~ "}" ^^ { fields =>
        TStruct(fields)
      }

  def solr_named_args: Parser[Array[(String, Map[String, AnyRef], AST)]] =
    repsep(solr_named_arg, ",") ^^ (_.toArray)

  def solr_field_spec: Parser[Map[String, AnyRef]] =
    "{" ~> repsep(solr_field_spec1, ",") <~ "}" ^^ (_.toMap)

  def solr_field_spec1: Parser[(String, AnyRef)] =
    (identifier <~ "=") ~ solr_literal ^^ { case id ~ v => (id, v) }

  def solr_literal: Parser[AnyRef] =
    "true" ^^ { _ => true.asInstanceOf[AnyRef] } |
      "false" ^^ { _ => false.asInstanceOf[AnyRef] } |
      stringLiteral ^^ (_.asInstanceOf[AnyRef])

  def solr_named_arg: Parser[(String, Map[String, AnyRef], AST)] =
    identifier ~ opt(solr_field_spec) ~ ("=" ~> expr) ^^ { case id ~ spec ~ expr => (id, spec.getOrElse(Map.empty), expr) }

  def parseSolrNamedArgs(code: String, ec: EvalContext): Array[(String, Map[String, AnyRef], Type, () => Option[Any])] = {
    val args = parseAll(solr_named_args, code) match {
      case Success(result, _) => result.asInstanceOf[Array[(String, Map[String, AnyRef], AST)]]
      case NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
    }
    args.map { case (id, spec, ast) =>
      ast.typecheck(ec)
      val t = ast.`type` match {
        case t: Type => t
        case _ => fatal(
          s"""invalid export expression resulting in unprintable type `${ ast.`type` }'""".stripMargin)
      }
      val f = ast.eval(ec)
      (id, spec, t, () => Option(f()))
    }
  }
}
