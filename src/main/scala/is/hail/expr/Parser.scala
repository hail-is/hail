package is.hail.expr

import is.hail.expr.types._
import is.hail.rvd.OrderedRVDType
import is.hail.utils.StringEscapeUtils._
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.sql.Row

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.parsing.combinator.JavaTokenParsers
import scala.util.parsing.input.Position

class RichParser[T](parser: Parser.Parser[T]) {
  def parse(input: String): T = {
    Parser.parseAll(parser, input) match {
      case Parser.Success(result, _) => result
      case Parser.NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
    }
  }

  def parseOpt(input: String): Option[T] = {
    Parser.parseAll(parser, input) match {
      case Parser.Success(result, _) => Some(result)
      case Parser.NoSuccess(msg, next) => None
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
      .map { case (name, (i, typ)) => (name, typ, i) }
    val f = t.compile().runWithDelayedValues(typedNames, ec)

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
    if (!t.isOfType(hr.typ))
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
    if (!t.isOfType(hr.typ))
      fatal(s"expression has wrong type: expected `${ hr.typ }', got $t")

    () => f().asInstanceOf[T]
  }

  def parseExprs(code: String, ec: EvalContext): (Array[Type], () => Array[Any]) = {
    val (types, fs) = args.parse(code).map(eval(_, ec)).unzip
    (types.toArray, () => fs.map(f => f()).toArray)
  }

  def parseSelectExprs(codes: Array[String], ec: EvalContext): (Array[Either[String, List[String]]], Array[Type], () => Array[Any]) = {

    // Left is string assignment, right is referenced path
    val names = new Array[Either[String, List[String]]](codes.length)
    val types = new Array[Type](codes.length)
    val fs = new Array[() => Any](codes.length)

    codes.indices.foreach { i =>
      val code = codes(i)
      annotationIdentifier.parseOpt(code) match {
        case Some(ids) =>
          val ast = expr.parse(code)
          assert(ids.nonEmpty)
          names(i) = Right(ids)
          val (t, f) = eval(ast, ec)
          types(i) = t
          fs(i) = f
        case None =>
          val (name, ast) = named_expr(identifier).parse(code)
          names(i) = Left(name)
          val (t, f) = eval(ast, ec)
          types(i) = t
          fs(i) = f
      }
    }
    (names, types, () => fs.map(_()))
  }

  def parseAnnotationExprs(code: String, ec: EvalContext, expectedHead: Option[String]): (
    Array[List[String]], Array[Type], () => Array[Any]) = {

    val (prePaths, ts, f) = parseNamedExprs(code, annotationIdentifier, ec)
    (prePaths.map { path =>
      expectedHead match {
        case Some(h) =>
          require(path.head == h)
          path.tail
        case None => path
      }
    }, ts, f)
  }

  def parseAnnotationExprsToAST(code: String, ec: EvalContext): Array[(String, AST)] = {
    val as = named_exprs(identifier).parse(code)
    as.foreach { case (_, ast) => ast.typecheck(ec) }
    as
  }

  def parseAnnotationExprsToAST(code: String, ec: EvalContext, expectedHead: Option[String]): Array[(String, AST)] = {
    named_exprs(annotationIdentifier)
      .parse(code).map { case (p, ast) =>
      ast.typecheck(ec)
      val n = expectedHead match {
        case Some(h) =>
          require(p.head == h && p.length == 2)
          p.last
        case None =>
          require(p.length == 1)
          p.head
      }
      (n, ast)
    }
  }


  def parseNamedExprs(code: String, ec: EvalContext): (Array[String], Array[Type], () => Array[Any]) = {
    val (names, types, f) = parseNamedExprs[String](code, identifier, ec)

    (names, types, f)
  }

  def parseNamedExprs[T](code: String, name: Parser[T], ec: EvalContext)(implicit ct: ClassTag[T]): (
    Array[T], Array[Type], () => Array[Any]) = {

    val parsed = named_exprs(name).parse(code)
    val nExprs = parsed.length

    val names = parsed.map(_._1)
    val asts = parsed.map(_._2)
    val tf = asts.map(eval(_, ec))
    val types = tf.map(_._1)
    val fs = tf.map(_._2)
    val f = () => fs.map(_())

    (names, types, f)
  }

  def parse[T](parser: Parser[T], code: String): T = {
    parseAll(parser, code) match {
      case Success(result, _) => result
      case NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
    }
  }

  def parseType(code: String): Type = parse(type_expr, code)

  def parseStructType(code: String): TStruct = parse(struct_expr, code)

  def parseOrderedRVDType(code: String): OrderedRVDType = parse(ordered_rvd_type_expr, code)

  def parseTableType(code: String): TableType = parse(table_type_expr, code)

  def parseMatrixType(code: String): MatrixType = parse(matrix_type_expr, code)

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

  def parseLocusInterval(input: String, rg: RGBase): Interval = {
    parseAll[Interval](locusInterval(rg), input) match {
      case Success(r, _) => r
      case NoSuccess(msg, next) => fatal(s"invalid interval expression: `$input': $msg")
    }
  }

  def parseCall(input: String): Call = {
    parseAll[Call](call, input) match {
      case Success(r, _) => r
      case NoSuccess(msg, next) => fatal(s"invalid call expression: `$input': $msg")
    }
  }

  def withPos[T](p: => Parser[T]): Parser[Positioned[T]] =
    positioned[Positioned[T]](p ^^ { x => Positioned(x) })

  def oneOfLiteral(s: String*): Parser[String] = oneOfLiteral(s.toArray)

  def oneOfLiteral(a: Array[String]): Parser[String] = new Parser[String] {
    var hasEnd: Boolean = false

    val m = a.flatMap { s =>
      val l = s.length
      if (l == 0) {
        hasEnd = true
        None
      }
      else if (l == 1) {
        Some((s.charAt(0), ""))
      }
      else
        Some((s.charAt(0), s.substring(1)))
    }.groupBy(_._1).mapValues { v => oneOfLiteral(v.map(_._2)) }

    def apply(in: Input): ParseResult[String] = {
      m.get(in.first) match {
        case Some(p) =>
          p(in.rest) match {
            case s: Success[_] =>
              Success(in.first.toString + s.result, in.drop(s.result.length + 1))
            case _ => Failure("", in)
          }
        case None =>
          if (hasEnd)
            Success("", in)
          else
            Failure("", in)
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

  def comma_delimited_doubles: Parser[Array[Double]] =
    repsep(floatingPointNumber, ",") ^^ (_.map(_.toDouble).toArray)

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
    rep(withPos("-" | "+" | "!")) ~ exponent_expr ^^ { case lst ~ rhs =>
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
    withPos("f32#" ~> """-?\d+(\.\d+)?[eE][+-]?\d+""".r) ^^ (r => Const(r.pos, r.x.toFloat, TFloat32())) |
    withPos("f64#" ~> """-?\d+(\.\d+)?[eE][+-]?\d+""".r) ^^ (r => Const(r.pos, r.x.toDouble, TFloat64())) |
      withPos("f32#" ~> """-?\d*(\.\d+)?""".r) ^^ (r => Const(r.pos, r.x.toFloat, TFloat32())) |
      withPos("f64#" ~> """-?\d*(\.\d+)?""".r) ^^ (r => Const(r.pos, r.x.toDouble, TFloat64())) |
      withPos("""-?\d+(\.\d+)?[eE][+-]?\d+[dD]?""".r) ^^ (r => Const(r.pos, r.x.toDouble, TFloat64())) |
      withPos("""-?\d*\.\d+[dD]?""".r) ^^ (r => Const(r.pos, r.x.toDouble, TFloat64())) |
      withPos("i64#" ~> wholeNumber) ^^ (r => Const(r.pos, r.x.toLong, TInt64())) |
      withPos("i32#" ~> wholeNumber) ^^ (r => Const(r.pos, r.x.toInt, TInt32())) |
      withPos(wholeNumber <~ "[Ll]".r) ^^ (r => Const(r.pos, r.x.toLong, TInt64())) |
      withPos(wholeNumber) ^^ (r => Const(r.pos, r.x.toInt, TInt32())) |
      withPos(stringLiteral) ^^ { r => Const(r.pos, r.x, TString()) } |
      withPos("NA" ~> ":" ~> type_expr) ^^ (r => Const(r.pos, null, r.x)) |
      withPos(arrayDeclaration) ^^ (r => ArrayConstructor(r.pos, r.x)) |
      withPos(structDeclaration) ^^ (r => StructConstructor(r.pos, r.x.map(_._1), r.x.map(_._2))) |
      withPos(tupleDeclaration) ^^ (r => TupleConstructor(r.pos, r.x)) |
      withPos("true") ^^ (r => Const(r.pos, true, TBoolean())) |
      withPos("false") ^^ (r => Const(r.pos, false, TBoolean())) |
      genomeReferenceDependentTypes ~ ("(" ~> identifier <~ ")") ~ withPos("(") ~ (args <~ ")") ^^ {
        case fn ~ rg ~ lparen ~ args => ReferenceGenomeDependentConstructor(lparen.pos, fn, rg, args)
      } |
      genomeReferenceDependentTypes ~ withPos("(") ~ (args <~ ")") ^^ {
        case fn ~ lparen ~ args => ReferenceGenomeDependentConstructor(lparen.pos, fn, ReferenceGenome.defaultReference.name, args)
      } |
      (guard(not("if" | "else")) ~> identifier) ~ withPos("(") ~ (args <~ ")") ^^ {
        case id ~ lparen ~ args =>
          Apply(lparen.pos, id, args)
      } |
      guard(not("if" | "else")) ~> withPos(identifier) ^^ (r => SymRef(r.pos, r.x)) |
      "{" ~> expr <~ "}" |
      "(" ~> expr <~ ")"

  def annotationSignature: Parser[TStruct] =
    type_fields ^^ { fields => TStruct(fields) }

  def arrayDeclaration: Parser[Array[AST]] = "[" ~> repsep(expr, ",") <~ "]" ^^ (_.toArray)

  def structDeclaration: Parser[Array[(String, AST)]] = "{" ~> repsep(structField, ",") <~ "}" ^^ (_.toArray)

  def tupleDeclaration: Parser[Array[AST]] = "Tuple(" ~> repsep(expr, ",") <~ ")" ^^ (_.toArray)

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

  override def stringLiteral: Parser[String] =
    quotedLiteral('"', "string literal") | quotedLiteral('\'', "string literal")

  def tuplify[T, S](p: ~[T, S]): (T, S) = p match {
    case t ~ s => (t, s)
  }

  def tuplify[T, S, V](p: ~[~[T, S], V]): (T, S, V) = p match {
    case t ~ s ~ v => (t, s, v)
  }

  def named_expr[T](name: Parser[T]): Parser[(T, AST)] =
    (name <~ "=") ~ expr ^^ { case n ~ e => n -> e }

  def named_exprs[T](name: Parser[T]): Parser[Array[(T, AST)]] =
    repsep(named_expr(name), ",") ^^ (_.toArray)

  def decorator: Parser[(String, String)] =
    ("@" ~> (identifier <~ "=")) ~ stringLiteral ^^ { case name ~ desc =>
      //    ("@" ~> (identifier <~ "=")) ~ stringLiteral("\"" ~> "[^\"]".r <~ "\"") ^^ { case name ~ desc =>
      (name, desc)
    }

  def type_field_decorator: Parser[(String, Type)] =
    (identifier <~ ":") ~ type_expr ~ rep(decorator) ^^ { case name ~ t ~ decorators => (name, t) }

  def type_field: Parser[(String, Type)] =
    (identifier <~ ":") ~ type_expr ^^ { case name ~ t => (name, t) }

  def type_fields: Parser[Array[Field]] = repsep(type_field_decorator | type_field, ",") ^^ {
    _.zipWithIndex.map { case ((id, t), index) => Field(id, t, index) }
      .toArray
  }

  def type_expr: Parser[Type] = _required_type ~ _type_expr ^^ { case req ~ t => t.setRequired(req) }

  def _required_type: Parser[Boolean] = "+" ^^ { _ => true } | success(false)

  def _type_expr: Parser[Type] =
    "Interval" ~> "[" ~> type_expr <~ "]" ^^ { pointType => TInterval(pointType) } |
      "Boolean" ^^ { _ => TBoolean() } |
      "Int32" ^^ { _ => TInt32() } |
      "Int64" ^^ { _ => TInt64() } |
      "Int" ^^ { _ => TInt32() } |
      "Float32" ^^ { _ => TFloat32() } |
      "Float64" ^^ { _ => TFloat64() } |
      "Float" ^^ { _ => TFloat64() } |
      "String" ^^ { _ => TString() } |
      "AltAllele" ^^ { _ => TAltAllele() } |
      ("Variant" ~ "(") ~> identifier <~ ")" ^^ { id => ReferenceGenome.getReference(id).variantType } |
      ("Locus" ~ "(") ~> identifier <~ ")" ^^ { id => ReferenceGenome.getReference(id).locusType } |
      ("LocusAlleles" ~ "(") ~> identifier <~ ")" ^^ { id => ReferenceGenome.getReference(id).locusType } |
      "Call" ^^ { _ => TCall() } |
      ("Array" ~ "[") ~> type_expr <~ "]" ^^ { elementType => TArray(elementType) } |
      ("Set" ~ "[") ~> type_expr <~ "]" ^^ { elementType => TSet(elementType) } |
      ("Dict" ~ "[") ~> type_expr ~ "," ~ type_expr <~ "]" ^^ { case kt ~ _ ~ vt => TDict(kt, vt) } |
      ("Tuple" ~ "[") ~> repsep(type_expr, ",") <~ "]" ^^ { types => TTuple(types.toArray) } |
      _struct_expr

  def struct_expr: Parser[TStruct] = _required_type ~ _struct_expr ^^ { case req ~ t => t.setRequired(req).asInstanceOf[TStruct] }

  def _struct_expr: Parser[TStruct] = ("Struct" ~ "{") ~> type_fields <~ "}" ^^ { fields => TStruct(fields) }

  def key: Parser[Array[String]] = "[" ~> (repsep(identifier, ",") ^^ { _.toArray }) <~ "]"

  def trailing_keys: Parser[Array[String]] = rep("," ~> identifier) ^^ { _.toArray }

  def ordered_rvd_type_expr: Parser[OrderedRVDType] =
    (("OrderedRVDType" ~ "{" ~ "key" ~ ":" ~ "[") ~> key) ~ (trailing_keys <~ "]") ~
      (("," ~ "row" ~ ":") ~> struct_expr <~ "}") ^^ { case partitionKey ~ restKey ~ rowType =>
      new OrderedRVDType(partitionKey, partitionKey ++ restKey, rowType)
    }

  def table_type_expr: Parser[TableType] =
    (("Table" ~ "{" ~ "global" ~ ":") ~> struct_expr) ~
      (("," ~ "key" ~ ":") ~> key) ~
      (("," ~ "row" ~ ":") ~> struct_expr <~ "}") ^^ { case globalType ~ key ~ rowType =>
      TableType(rowType, key, globalType)
    }

  def matrix_type_expr: Parser[MatrixType] =
    (("Matrix" ~ "{" ~ "global" ~ ":") ~> struct_expr) ~
      (("," ~ "col_key" ~ ":") ~> key) ~
      (("," ~ "col" ~ ":") ~> struct_expr) ~
      (("," ~ "row_key" ~ ":" ~ "[") ~> key) ~ (trailing_keys <~ "]") ~
      (("," ~ "row" ~ ":") ~> struct_expr) ~
      (("," ~ "entry" ~ ":") ~> struct_expr <~ "}") ^^ { case globalType ~ colKey ~ colType ~ rowPartitionKey ~ rowRestKey ~ rowType ~ entryType =>
      MatrixType.fromParts(globalType, colKey, colType, rowPartitionKey, rowPartitionKey ++ rowRestKey, rowType, entryType)
    }

  def parsePhysicalType(code: String): PhysicalType = parse(physical_type, code)

  def physical_type: Parser[PhysicalType] =
    ("Default" ~ "[") ~> type_expr <~ "]" ^^ { t => PDefault(t) }

  def parseEncodedType(code: String): PhysicalType = parse(physical_type, code)

  def encoded_type: Parser[EncodedType] =
    ("Default" ~ "[") ~> type_expr <~ "]" ^^ { t => EDefault(t) }

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

  def call: Parser[Call] = {
    wholeNumber ~ "/" ~ rep1sep(wholeNumber, "/") ^^ { case a0 ~ _ ~ arest =>
      CallN(coerceInt(a0) +: arest.map(coerceInt).toArray, phased = false) } |
      wholeNumber ~ "|" ~ rep1sep(wholeNumber, "|") ^^ { case a0 ~ _ ~ arest =>
        CallN(coerceInt(a0) +: arest.map(coerceInt).toArray, phased = true) } |
      wholeNumber ^^ { a => Call1(coerceInt(a), phased = false) } |
      "|" ~ wholeNumber ^^ { case _ ~ a => Call1(coerceInt(a), phased = true) } |
      "-" ^^ { _ => Call0(phased = false) } |
      "|-" ^^ { _ => Call0(phased = true) }
  }

  def genomeReferenceDependentTypes: Parser[String] = "Variant" | "LocusInterval" | "LocusAlleles" | "Locus"

  def intervalWithEndpoints[T](bounds: Parser[(T, T)]): Parser[Interval] = {
    val start = ("[" ^^^ true) | ("(" ^^^ false)
    val end = ("]" ^^^ true) | (")" ^^^ false)

    start ~ bounds ~ end ^^ {case istart ~ int ~ iend => Interval(int._1, int._2, istart, iend)} |
      bounds ^^ {int => Interval(int._1, int._2, true, false)}
  }

  def locusInterval(rg: RGBase): Parser[Interval] = {
    val contig = rg.contigParser
    val valueParser = locus(rg) ~ "-" ~ locus(rg) ^^ { case l1 ~ _ ~ l2 => (l1, l2) } |
      locus(rg) ~ "-" ~ pos ^^ { case l1 ~ _ ~ p2 => (l1, l1.copyChecked(rg, position = p2.getOrElse(rg.contigLength(l1.contig)))) } |
      contig ~ "-" ~ contig ^^ { case c1 ~ _ ~ c2 => (Locus(c1, 1, rg), Locus(c2, rg.contigLength(c2), rg)) } |
      contig ^^ { c => (Locus(c, 1), Locus(c, rg.contigLength(c))) }
      intervalWithEndpoints(valueParser)
  }

  def locus(rg: RGBase): Parser[Locus] =
    (rg.contigParser ~ ":" ~ pos) ^^ { case c ~ _ ~ p => Locus(c, p.getOrElse(rg.contigLength(c)), rg) }

  def coerceInt(s: String): Int = try {
    s.toInt
  } catch {
    case e: java.lang.NumberFormatException => Int.MaxValue
  }

  def exp10(i: Int): Int = {
    var mult = 1
    var j = 0
    while (j < i) {
      mult *= 10
      j += 1
    }
    mult
  }

  def pos: Parser[Option[Int]] = {
    "[sS][Tt][Aa][Rr][Tt]".r ^^ { _ => Some(1) } |
      "[Ee][Nn][Dd]".r ^^ { _ => None } |
      "\\d+".r <~ "[Kk]".r ^^ { i => Some(coerceInt(i) * 1000) } |
      "\\d+".r <~ "[Mm]".r ^^ { i => Some(coerceInt(i) * 1000000) } |
      "\\d+".r ~ "." ~ "\\d{1,3}".r ~ "[Kk]".r ^^ { case lft ~ _ ~ rt ~ _ => Some(coerceInt(lft + rt) * exp10(3 - rt.length)) } |
      "\\d+".r ~ "." ~ "\\d{1,6}".r ~ "[Mm]".r ^^ { case lft ~ _ ~ rt ~ _ => Some(coerceInt(lft + rt) * exp10(6 - rt.length)) } |
      "\\d+".r ^^ { i => Some(coerceInt(i)) }
  }
}
