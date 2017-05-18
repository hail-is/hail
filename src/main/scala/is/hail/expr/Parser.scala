package is.hail.expr

import is.hail.utils.StringEscapeUtils._
import is.hail.utils._
import org.apache.spark.sql.Row

import scala.collection.mutable
import scala.util.parsing.combinator.JavaTokenParsers
import scala.util.parsing.input.Position

class RichParser[T](parser: Parser.Parser[T]) {
  def parse(input: String): T = {
    Parser.parseAll(parser, input) match {
      case Parser.Success(result, _) => result
      case Parser.NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
    }
  }
}

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

  def error(pos: Position, msg: String, tr: Truncatable): Nothing = {
    val lineContents = pos.longString.split("\n").head
    val prefix = s"<input>:${ pos.line }:"
    fatal(
      s"""$msg
         |$prefix$lineContents
         |${ " " * prefix.length }${
        lineContents.take(pos.column - 1).map { c => if (c == '\t') c else ' ' }
      }^""".stripMargin, tr)
  }
}

object Parser extends JavaTokenParsers {
  private def evalNoTypeCheck(t: AST, ec: EvalContext): () => Any = {
    val typedNames = ec.st.toSeq
      .sortBy { case (name, (i, _)) => i }
      .map { case (name, (_, typ)) => (name, typ) }
    val f = t.compile().runWithDelayedValues(typedNames.toSeq, ec)

    // FIXME: ec.a is actually mutable.ArrayBuffer[AnyRef] because Annotation is
    // actually AnyRef, but there's a lot to change
    () => f(ec.a.asInstanceOf[mutable.ArrayBuffer[AnyRef]])
  }

  private def eval(t: AST, ec: EvalContext): (Type, () => Any) = {
    t.typecheck(ec)

    if (!t.`type`.isRealizable)
      t.parseError(s"unrealizable type `${ t.`type` }' as result of expression")

    val thunk = evalNoTypeCheck(t, ec)
    (t.`type`, thunk)
  }

  def evalTypedExpr[T](ast: AST, ec: EvalContext)(implicit hr: HailRep[T]): () => T = {
    val (t, f) = evalExpr(ast, ec)
    if (t != hr.typ)
      fatal(s"expression has wrong type: expected `${ hr.typ }', got $t")

    () => f().asInstanceOf[T]
  }

  def evalExpr(ast: AST, ec: EvalContext): (Type, () => Any) = eval(ast, ec)

  def parseExpr(code: String, ec: EvalContext): (Type, () => Any) = {
    eval(expr.parse(code), ec)
  }

  def parseToAST(code: String, ec: EvalContext): AST = {
    val t = expr.parse(code)
    t.typecheck(ec)
    t
  }

  def parseTypedExpr[T](code: String, ec: EvalContext)(implicit hr: HailRep[T]): () => T = {
    val (t, f) = parseExpr(code, ec)
    if (t != hr.typ)
      fatal(s"expression has wrong type: expected `${ hr.typ }', got $t")

    () => f().asInstanceOf[T]
  }

  def parseExprs(code: String, ec: EvalContext): (Array[Type], () => Array[Any]) = {
    val (types, fs) = args.parse(code).map(eval(_, ec)).unzip
    (types.toArray, () => fs.map(f => f()).toArray)
  }

  def parseAnnotationExprs(code: String, ec: EvalContext, expectedHead: Option[String]): (
    Array[List[String]], Array[Type], () => Array[Any]) = {
    val (maybeNames, types, f) = parseNamedExprs[List[String]](code, annotationIdentifier, ec,
      (t, s) => t.map(_ :+ s))

    if (maybeNames.exists(_.isEmpty))
      fatal("left-hand side required in annotation expression")

    val names = maybeNames.map(_.get)

    expectedHead.foreach { h =>
      names.foreach { n =>
        if (n.head != h)
          fatal(
            s"""invalid annotation path `${ n.map(prettyIdentifier).mkString(".") }'
               |  Path should begin with `$h'
           """.stripMargin)
      }
    }

    (names.map { n =>
      if (expectedHead.isDefined)
        n.tail
      else
        n
    }, types, () => {
      f()
    })
  }

  def parseExportExprs(code: String, ec: EvalContext): (Option[Array[String]], Array[Type], () => Array[String]) = {
    val (names, types, f) = parseNamedExprs[String](code, tsvIdentifier, ec,
      (t, s) => Some(t.map(_ + "." + s).getOrElse(s)))

    val allNamed = names.forall(_.isDefined)
    val noneNamed = names.forall(_.isEmpty)

    if (!allNamed && !noneNamed)
      fatal(
        """export expressions require either all arguments named or none
          |  Hint: exploded structs (e.g. va.info.*) count as named arguments""".stripMargin)

    (anyFailAllFail(names), types,
      () => {
        (types, f()).zipped.map { case (t, v) =>
          t.str(v)
        }
      })
  }

  def parseNamedExprs(code: String, ec: EvalContext): (Array[String], Array[Type], () => Array[Any]) = {
    val (maybeNames, types, f) = parseNamedExprs[String](code, identifier, ec,
      (t, s) => Some(t.map(_ + "." + s).getOrElse(s)))

    if (maybeNames.exists(_.isEmpty))
      fatal("left-hand side required in named expression")

    val names = maybeNames.map(_.get)

    (names, types, f)
  }

  def parseNamedExprs[T](code: String, name: Parser[T], ec: EvalContext, concat: (Option[T], String) => Option[T]): (
    Array[Option[T]], Array[Type], () => Array[Any]) = {

    val parsed = named_exprs(name).parse(code)
    val nExprs = parsed.size

    val nValues = parsed.map { case (n, ast, splat) =>
      ast.typecheck(ec)
      if (splat) {
        ast.`type` match {
          case t: TStruct =>
            t.size

          case t =>
            fatal(s"cannot splat non-struct type: $t")
        }
      } else
        1
    }.sum

    val a = new Array[Any](nValues)

    val names = new Array[Option[T]](nValues)
    val types = new Array[Type](nValues)
    val fs = new Array[() => Unit](nExprs)

    var i = 0
    var j = 0
    parsed.foreach { case (n, ast, splat) =>
      val t = ast.`type`

      if (!t.isRealizable)
        fatal(s"unrealizable type in export expression: $t")

      val f = evalNoTypeCheck(ast, ec)
      if (splat) {
        val j0 = j
        val s = t.asInstanceOf[TStruct] // checked above
        s.fields.foreach { field =>
          names(j) = concat(n, field.name)
          types(j) = field.typ
          j += 1
        }

        val sSize = s.size
        fs(i) = () => {
          val v = f()
          if (v == null) {
            var k = 0
            while (k < sSize) {
              a(j0 + k) = null
              k += 1
            }
          } else {
            val va = v.asInstanceOf[Row].toSeq.toArray[Any]
            var k = 0
            while (k < sSize) {
              a(k + j0) = va(k)
              k += 1
            }
          }
        }
      } else {
        names(j) = n
        types(j) = t
        val localJ = j
        fs(i) = () => {
          a(localJ) = f()
        }
        j += 1
      }

      i += 1
    }
    assert(i == nExprs)
    assert(j == nValues)

    (names, types, () => {
      fs.foreach(_ ())

      val newa = new Array[Any](nValues)
      System.arraycopy(a, 0, newa, 0, nValues)
      newa
    })
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
        case Success(result, _) => result.map(f => (f.name, f.typ)).toMap
        case NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
      }
  }

  def parseIdentifierList(code: String): Array[String] = {
    parseAll(identifierList, code) match {
      case Success(result, _) => result
      case NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
    }
  }

  def parseCommaDelimitedDoubles(code: String): Array[Double] = {
    parseAll(comma_delimited_doubles, code) match {
      case Success(r, _) => r
      case NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
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
      fatal(s"expected an annotation path starting in `$root', but got a path starting in '${ path.head }'")
    else
      path.tail
  }

  def validateAnnotationRoot(a: String, root: String): Unit = {
    parseAnnotationRoot(a, root)
  }

  def withPos[T](p: => Parser[T]): Parser[Positioned[T]] =
    positioned[Positioned[T]](p ^^ { x => Positioned(x) })

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
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => Apply(op.pos, op.x, Array(acc, rhs)) }
    }

  def and_expr: Parser[AST] =
    lt_expr ~ rep(withPos("&&" | "&") ~ lt_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => Apply(op.pos, op.x, Array(acc, rhs)) }
    }

  def lt_expr: Parser[AST] =
    eq_expr ~ rep(withPos("<=" | ">=" | "<" | ">") ~ eq_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => Apply(op.pos, op.x, Array(acc, rhs)) }
    }

  def eq_expr: Parser[AST] =
    add_expr ~ rep(withPos("==" | "!=") ~ add_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => Apply(op.pos, op.x, Array(acc, rhs)) }
    }

  def add_expr: Parser[AST] =
    mul_expr ~ rep(withPos("+" | "-") ~ mul_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => Apply(op.pos, op.x, Array(acc, rhs)) }
    }

  def mul_expr: Parser[AST] =
    tilde_expr ~ rep(withPos("*" | "//" | "/" | "%") ~ tilde_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => Apply(op.pos, op.x, Array(acc, rhs)) }
    }

  def tilde_expr: Parser[AST] =
    unary_expr ~ rep(withPos("~") ~ unary_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => Apply(op.pos, op.x, Array(acc, rhs)) }
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
    repsep(annotationExpression, ",") ^^ {
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

  def identifierList: Parser[Array[String]] = repsep(identifier, ",") ^^ {
    _.toArray
  }

  def args: Parser[Array[AST]] =
    repsep(expr, ",") ^^ {
      _.toArray
    }

  def unary_expr: Parser[AST] =
    rep(withPos("-" | "!" | "**")) ~ exponent_expr ^^ { case lst ~ rhs =>
      lst.foldRight(rhs) { case (op, acc) =>
        Apply(op.pos, op.x, Array(acc))
      }
    }

  def exponent_expr: Parser[AST] =
    dot_expr ~ rep(withPos("**") ~ dot_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => Apply(op.pos, op.x, Array(acc, rhs)) }
    }

  def dot_expr: Parser[AST] =
    primary_expr ~ rep((withPos(".") ~ identifier ~ "(" ~ args ~ ")")
      | (withPos(".") ~ identifier)
      | withPos("[") ~ expr ~ "]"
      | withPos("[") ~ opt(expr) ~ ":" ~ opt(expr) ~ "]") ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { (acc, t) =>
        (t: @unchecked) match {
          case (dot: Positioned[_]) ~ sym => Select(dot.pos, acc, sym)
          case (dot: Positioned[_]) ~ (sym: String) ~ "(" ~ (args: Array[AST]) ~ ")" =>
            ApplyMethod(dot.pos, acc, sym, args)
          case (lbracket: Positioned[_]) ~ (idx: AST) ~ "]" => ApplyMethod(lbracket.pos, acc, "[]", Array(idx))
          case (lbracket: Positioned[_]) ~ None ~ ":" ~ None ~ "]" =>
            ApplyMethod(lbracket.pos, acc, "[:]", Array())
          case (lbracket: Positioned[_]) ~ Some(idx1: AST) ~ ":" ~ None ~ "]" =>
            ApplyMethod(lbracket.pos, acc, "[*:]", Array(idx1))
          case (lbracket: Positioned[_]) ~ None ~ ":" ~ Some(idx2: AST) ~ "]" =>
            ApplyMethod(lbracket.pos, acc, "[:*]", Array(idx2))
          case (lbracket: Positioned[_]) ~ Some(idx1: AST) ~ ":" ~ Some(idx2: AST) ~ "]" =>
            ApplyMethod(lbracket.pos, acc, "[*:*]", Array(idx1, idx2))
        }
      }
    }

  def primary_expr: Parser[AST] =
    withPos("""-?\d*\.\d+[dD]?""".r) ^^ (r => Const(r.pos, r.x.toDouble, TDouble)) |
      withPos("""-?\d+(\.\d*)?[eE][+-]?\d+[dD]?""".r) ^^ (r => Const(r.pos, r.x.toDouble, TDouble)) |
      withPos(wholeNumber <~ "[Ll]".r) ^^ (r => Const(r.pos, r.x.toLong, TLong)) |
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
              ParserUtils.error(s.pos, s"invalid $what")
          }

      }
    } | withPos(s"$delim([^$delim\\\\]|\\\\.)*\\z".r) ^^ { s =>
      ParserUtils.error(s.pos, s"unterminated $what")
    }

  def backtickLiteral: Parser[String] = quotedLiteral('`', "backtick identifier")

  override def stringLiteral: Parser[String]

  = quotedLiteral('"', "string literal")

  def tuplify[T, S](p: ~[T, S]): (T, S) = p match {
    case t ~ s => (t, s)
  }

  def tuplify[T, S, V](p: ~[~[T, S], V]): (T, S, V) = p match {
    case t ~ s ~ v => (t, s, v)
  }

  def splat: Parser[Boolean] =
    "." ~ "*" ^^ { _ => true } |
      success(false)

  def named_expr[T](name: Parser[T]): Parser[(Option[T], AST, Boolean)] =
    (((name <~ "=") ^^ { n => Some(n) }) ~ expr ~ splat |||
      success(None) ~ expr ~ splat) ^^ tuplify

  def named_exprs[T](name: Parser[T]): Parser[Seq[(Option[T], AST, Boolean)]] =
    repsep(named_expr(name), ",")

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
      "Char" ^^ { _ => TString } | // FIXME backward-compatibility, remove this at some point
      "Int" ^^ { _ => TInt } |
      "Long" ^^ { _ => TLong } |
      "Float" ^^ { _ => TFloat } |
      "Double" ^^ { _ => TDouble } |
      "String" ^^ { _ => TString } |
      "Sample" ^^ { _ => TString } | // FIXME back-compatibility
      "AltAllele" ^^ { _ => TAltAllele } |
      "Variant" ^^ { _ => TVariant } |
      "Locus" ^^ { _ => TLocus } |
      "Genotype" ^^ { _ => TGenotype } |
      "Call" ^^ { _ => TCall } |
      "String" ^^ { _ => TString } |
      ("Array" ~ "[") ~> type_expr <~ "]" ^^ { elementType => TArray(elementType) } |
      ("Set" ~ "[") ~> type_expr <~ "]" ^^ { elementType => TSet(elementType) } |
      // back compatibility
      ("Dict" ~ "[") ~> type_expr <~ "]" ^^ { elementType => TDict(TString, elementType) } |
      ("Dict" ~ "[") ~> type_expr ~ "," ~ type_expr <~ "]" ^^ { case kt ~ _ ~ vt => TDict(kt, vt) } |
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

  def parseSolrNamedArgs(code: String, ec: EvalContext): Array[(String, Map[String, AnyRef], Type, () => Any)] = {
    val args = parseAll(solr_named_args, code) match {
      case Success(result, _) => result.asInstanceOf[Array[(String, Map[String, AnyRef], AST)]]
      case NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
    }
    args.map { case (id, spec, ast) =>
      ast.typecheck(ec)
      val t = ast.`type`
      if (!t.isRealizable)
        fatal(s"unrealizable type in Solr export expression: $t")
      val f = evalNoTypeCheck(ast, ec)
      (id, spec, t, f)
    }
  }
}
