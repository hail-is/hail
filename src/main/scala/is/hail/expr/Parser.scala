package is.hail.expr

import is.hail.HailContext
import is.hail.annotations.BroadcastRow
import is.hail.expr.ir.{AggSignature, IR, MatrixIR, TableIR}
import is.hail.expr.types._
import is.hail.rvd.OrderedRVDType
import is.hail.table.{Ascending, Descending, SortField, TableSpec}
import is.hail.utils.StringEscapeUtils._
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.sql.Row
import org.json4s.jackson.JsonMethods

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
  def parseToAST(code: String, ec: EvalContext): AST = {
    val t = expr.parse(code)
    t.typecheck(ec)
    t
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
      IfAST(ifx.pos, cond, thenTree, elseTree)
    }

  def let_expr: Parser[AST] =
    withPos("let") ~ rep1sep((identifier <~ "=") ~ expr, "and") ~ ("in" ~> expr) ^^ { case let ~ bindings ~ body =>
      LetAST(let.pos, bindings.iterator.map { case id ~ v => (id, v) }.toArray, body)
    }

  def or_expr: Parser[AST] =
    and_expr ~ rep(withPos("||" | "|") ~ and_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => ApplyAST(op.pos, op.x, Array(acc, rhs)) }
    }

  def and_expr: Parser[AST] =
    lt_expr ~ rep(withPos("&&" | "&") ~ lt_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => ApplyAST(op.pos, op.x, Array(acc, rhs)) }
    }

  def lt_expr: Parser[AST] =
    eq_expr ~ rep(withPos("<=" | ">=" | "<" | ">") ~ eq_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => ApplyAST(op.pos, op.x, Array(acc, rhs)) }
    }

  def eq_expr: Parser[AST] =
    add_expr ~ rep(withPos("==" | "!=") ~ add_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => ApplyAST(op.pos, op.x, Array(acc, rhs)) }
    }

  def add_expr: Parser[AST] =
    mul_expr ~ rep(withPos("+" | "-") ~ mul_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => ApplyAST(op.pos, op.x, Array(acc, rhs)) }
    }

  def mul_expr: Parser[AST] =
    tilde_expr ~ rep(withPos("*" | "//" | "/" | "%") ~ tilde_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => ApplyAST(op.pos, op.x, Array(acc, rhs)) }
    }

  def tilde_expr: Parser[AST] =
    unary_expr ~ rep(withPos("~") ~ unary_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => ApplyAST(op.pos, op.x, Array(acc, rhs)) }
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
        ApplyAST(op.pos, op.x, Array(acc))
      }
    }

  def exponent_expr: Parser[AST] =
    dot_expr ~ rep(withPos("**") ~ dot_expr) ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { case (acc, op ~ rhs) => ApplyAST(op.pos, op.x, Array(acc, rhs)) }
    }

  def dot_expr: Parser[AST] =
    primary_expr ~ rep((withPos(".") ~ identifier ~ "(" ~ args ~ ")")
      | (withPos(".") ~ identifier)
      | withPos("[") ~ expr ~ "]"
      | withPos("[") ~ opt(expr) ~ ":" ~ opt(expr) ~ "]") ^^ { case lhs ~ lst =>
      lst.foldLeft(lhs) { (acc, t) =>
        (t: @unchecked) match {
          case (dot: Positioned[_]) ~ sym => SelectAST(dot.pos, acc, sym)
          case (dot: Positioned[_]) ~ (sym: String) ~ "(" ~ (args: Array[AST]) ~ ")" =>
            ApplyMethodAST(dot.pos, acc, sym, args)
          case (lbracket: Positioned[_]) ~ (idx: AST) ~ "]" => ApplyMethodAST(lbracket.pos, acc, "[]", Array(idx))
          case (lbracket: Positioned[_]) ~ None ~ ":" ~ None ~ "]" =>
            ApplyMethodAST(lbracket.pos, acc, "[:]", Array())
          case (lbracket: Positioned[_]) ~ Some(idx1: AST) ~ ":" ~ None ~ "]" =>
            ApplyMethodAST(lbracket.pos, acc, "[*:]", Array(idx1))
          case (lbracket: Positioned[_]) ~ None ~ ":" ~ Some(idx2: AST) ~ "]" =>
            ApplyMethodAST(lbracket.pos, acc, "[:*]", Array(idx2))
          case (lbracket: Positioned[_]) ~ Some(idx1: AST) ~ ":" ~ Some(idx2: AST) ~ "]" =>
            ApplyMethodAST(lbracket.pos, acc, "[*:*]", Array(idx1, idx2))
        }
      }
    }

  def primary_expr: Parser[AST] =
    withPos("f32#" ~> "nan") ^^ (r => Const(r.pos, Float.NaN, TFloat32())) |
      withPos("f32#" ~> "inf") ^^ (r => Const(r.pos, Float.PositiveInfinity, TFloat32())) |
      withPos("f32#" ~> "neginf") ^^ (r => Const(r.pos, Float.NegativeInfinity, TFloat32())) |
      withPos("f32#" ~> """-?\d+(\.\d+)?[eE][+-]?\d+""".r) ^^ (r => Const(r.pos, r.x.toFloat, TFloat32())) |
      withPos("f32#" ~> """-?\d*(\.\d+)?""".r) ^^ (r => Const(r.pos, r.x.toFloat, TFloat32())) |
      withPos("f64#" ~> "nan") ^^ (r => Const(r.pos, Double.NaN, TFloat64())) |
      withPos("f64#" ~> "inf") ^^ (r => Const(r.pos, Double.PositiveInfinity, TFloat64())) |
      withPos("f64#" ~> "neginf") ^^ (r => Const(r.pos, Double.NegativeInfinity, TFloat64())) |
      withPos("f64#" ~> """-?\d+(\.\d+)?[eE][+-]?\d+""".r) ^^ (r => Const(r.pos, r.x.toDouble, TFloat64())) |
      withPos("f64#" ~> """-?\d*(\.\d+)?""".r) ^^ (r => Const(r.pos, r.x.toDouble, TFloat64())) |
      withPos("""-?\d+(\.\d+)?[eE][+-]?\d+[fF]""".r) ^^ (r => Const(r.pos, r.x.toFloat, TFloat32())) |
      withPos("""-?\d*\.\d+[fF]""".r) ^^ (r => Const(r.pos, r.x.toFloat, TFloat32())) |
      withPos("""-?\d+[fF]""".r) ^^ (r => Const(r.pos, r.x.toFloat, TFloat32())) |
      withPos("""-?\d+[dD]""".r) ^^ (r => Const(r.pos, r.x.toDouble, TFloat64())) |
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
      referenceGenomeDependentFunction ~ ("(" ~> identifier <~ ")") ~ withPos("(") ~ (args <~ ")") ^^ {
        case fn ~ rg ~ lparen ~ args => ReferenceGenomeDependentFunction(lparen.pos, fn, rg, args)
      } |
      referenceGenomeDependentFunction ~ withPos("(") ~ (args <~ ")") ^^ {
        case fn ~ lparen ~ args => ReferenceGenomeDependentFunction(lparen.pos, fn, ReferenceGenome.defaultReference.name, args)
      } |
      (guard(not("if" | "else")) ~> identifier) ~ withPos("(") ~ (args <~ ")") ^^ {
        case id ~ lparen ~ args =>
          ApplyAST(lparen.pos, id, args)
      } |
      guard(not("if" | "else")) ~> withPos(identifier) ^^ (r => SymRefAST(r.pos, r.x)) |
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
    new Parser[String] {
      def apply(in: Input): ParseResult[String] = {
        var r = in

        val source = in.source
        val offset = in.offset
        val start = handleWhiteSpace(source, offset)
        r = r.drop(start - offset)

        if (r.atEnd || r.first != delim)
          return Failure(s"expected $what", r)
        r = r.rest

        val sb = new StringBuilder()

        val escapeChars = "\\bfnrt'\"`".toSet
        var continue = true
        while (continue) {
          if (r.atEnd)
            return Failure(s"unterminated $what", r)
          val c = r.first
          r = r.rest
          if (c == delim)
            continue = false
          else {
            sb += c
            if (c == '\\') {
              if (r.atEnd)
                return Failure(s"unterminated $what", r)
              val d = r.first
              if (!escapeChars.contains(d))
                return Failure(s"invalid escape character in $what", r)
              sb += d
              r = r.rest
            }
          }
        }
        Success(unescapeString(sb.result()), r)
      }
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

  def key: Parser[Array[String]] = "[" ~> (repsep(identifier, ",") ^^ {
    _.toArray
  }) <~ "]"

  def trailing_keys: Parser[Array[String]] = rep("," ~> identifier) ^^ {
    _.toArray
  }

  def ordered_rvd_type_expr: Parser[OrderedRVDType] =
    (("OrderedRVDType" ~ "{" ~ "key" ~ ":" ~ "[") ~> key) ~ (trailing_keys <~ "]") ~
      (("," ~ "row" ~ ":") ~> struct_expr <~ "}") ^^ { case partitionKey ~ restKey ~ rowType =>
      new OrderedRVDType(partitionKey, partitionKey ++ restKey, rowType)
    }

  def table_type_expr: Parser[TableType] =
    (("Table" ~ "{" ~ "global" ~ ":") ~> struct_expr) ~
      (("," ~ "key" ~ ":") ~> ("None" ^^ { _ => None } | key ^^ { key => Some(key.toIndexedSeq) })) ~
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

  def call: Parser[Call] = {
    wholeNumber ~ "/" ~ rep1sep(wholeNumber, "/") ^^ { case a0 ~ _ ~ arest =>
      CallN(coerceInt(a0) +: arest.map(coerceInt).toArray, phased = false)
    } |
      wholeNumber ~ "|" ~ rep1sep(wholeNumber, "|") ^^ { case a0 ~ _ ~ arest =>
        CallN(coerceInt(a0) +: arest.map(coerceInt).toArray, phased = true)
      } |
      wholeNumber ^^ { a => Call1(coerceInt(a), phased = false) } |
      "|" ~ wholeNumber ^^ { case _ ~ a => Call1(coerceInt(a), phased = true) } |
      "-" ^^ { _ => Call0(phased = false) } |
      "|-" ^^ { _ => Call0(phased = true) }
  }

  def referenceGenomeDependentFunction: Parser[String] = "LocusInterval" | "LocusAlleles" | "Locus" |
    "getReferenceSequence" | "isValidContig" | "isValidLocus" | "liftoverLocusInterval" | "liftoverLocus" |
    "locusToGlobalPos" | "globalPosToLocus"

  def intervalWithEndpoints[T](bounds: Parser[(T, T, Boolean, Boolean)]): Parser[Interval] = {
    val start = ("[" ^^^ true) | ("(" ^^^ false)
    val end = ("]" ^^^ true) | (")" ^^^ false)

    start ~ bounds ~ end ^^ { case istart ~ int ~ iend => Interval(int._1, int._2, istart, iend) } |
      bounds ^^ { int => Interval(int._1, int._2, int._3, int._4) }
  }

  def locusInterval(rgBase: RGBase): Parser[Interval] = {
    val rg = rgBase.asInstanceOf[ReferenceGenome]
    val contig = rg.contigParser

    val valueParser =
      locusUnchecked(rg) ~ "-" ~ rg.contigParser ~ ":" ~ pos ^^ { case l1 ~ _ ~ c2 ~ _ ~ p2 => p2 match {
        case Some(p) => (l1, Locus(c2, p), true, false)
        case None => (l1, Locus(c2, rg.contigLength(c2)), true, true)
      }
      } |
        locusUnchecked(rg) ~ "-" ~ pos ^^ { case l1 ~ _ ~ p2 => p2 match {
          case Some(p) => (l1, l1.copy(position = p), true, false)
          case None => (l1, l1.copy(position = rg.contigLength(l1.contig)), true, true)
        }
        } |
        contig ~ "-" ~ contig ^^ { case c1 ~ _ ~ c2 => (Locus(c1, 1), Locus(c2, rg.contigLength(c2)), true, true) } |
        contig ^^ { c => (Locus(c, 1), Locus(c, rg.contigLength(c)), true, true) }

    intervalWithEndpoints(valueParser) ^^ { i =>
      val normInterval = rg.normalizeLocusInterval(i)
      rg.checkLocusInterval(normInterval)
      normInterval
    }
  }

  def locusUnchecked(rg: RGBase): Parser[Locus] =
    (rg.contigParser ~ ":" ~ pos) ^^ { case c ~ _ ~ p => Locus(c, p.getOrElse(rg.contigLength(c))) }

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

  def int32_literal: Parser[Int] = wholeNumber.map(_.toInt)

  def int64_literal: Parser[Long] = wholeNumber.map(_.toLong)

  def float32_literal: Parser[Float] =
    "nan" ^^ { _ => Float.NaN } |
      "inf" ^^ { _ => Float.PositiveInfinity } |
      "neginf" ^^ { _ => Float.NegativeInfinity } |
      """-?\d+(\.\d+)?[eE][+-]?\d+""".r ^^ {
        _.toFloat
      } |
      """-?\d*(\.\d+)?""".r ^^ {
        _.toFloat
      }

  def float64_literal: Parser[Double] =
    "nan" ^^ { _ => Double.NaN } |
      "inf" ^^ { _ => Double.PositiveInfinity } |
      "neginf" ^^ { _ => Double.NegativeInfinity } |
      """-?\d+(\.\d+)?[eE][+-]?\d+""".r ^^ {
        _.toDouble
      } |
      """-?\d*(\.\d+)?""".r ^^ {
        _.toDouble
      }

  def int32_literals: Parser[IndexedSeq[Int]] = "(" ~> rep(int32_literal) <~ ")" ^^ {
    _.toFastIndexedSeq
  }

  def int32_literal_opt: Parser[Option[Int]] = int32_literal ^^ {
    Some(_)
  } | "None" ^^ { _ => None }

  def int64_literals: Parser[IndexedSeq[Long]] = "(" ~> rep(int64_literal) <~ ")" ^^ {
    _.toFastIndexedSeq
  }

  def int64_literals_opt: Parser[Option[IndexedSeq[Long]]] = int64_literals ^^ {
    Some(_)
  } | "None" ^^ { _ => None }

  def string_literal: Parser[String] = stringLiteral

  def string_literals: Parser[IndexedSeq[String]] = "(" ~> rep(string_literal).map(_.toFastIndexedSeq) <~ ")"

  def string_literals_opt: Parser[Option[IndexedSeq[String]]] = string_literals ^^ {
    Some(_)
  } | "None" ^^ { _ => None }

  def boolean_literal: Parser[Boolean] = "True" ^^ { _ => true } | "False" ^^ { _ => false }

  def ir_identifier: Parser[String] = identifier

  def ir_identifiers: Parser[IndexedSeq[String]] = "(" ~> rep(ir_identifier) <~ ")" ^^ {
    _.toFastIndexedSeq
  }

  def ir_binary_op: Parser[ir.BinaryOp] =
    ir_identifier ^^ { x => ir.BinaryOp.fromString(x) }

  def ir_unary_op: Parser[ir.UnaryOp] =
    ir_identifier ^^ { x => ir.UnaryOp.fromString(x) }

  def ir_comparison_op: Parser[ir.ComparisonOp] =
    "(" ~> ir_identifier ~ type_expr ~ type_expr <~ ")" ^^ { case x ~ t1 ~ t2 => ir.ComparisonOp.fromStringAndTypes(x, t1, t2) }

  def ir_agg_op: Parser[ir.AggOp] =
    ir_identifier ^^ { x => ir.AggOp.fromString(x) }

  def ir_children: Parser[IndexedSeq[ir.IR]] = rep(ir_value_expr) ^^ {
    _.toFastIndexedSeq
  }

  def table_ir_children: Parser[IndexedSeq[ir.TableIR]] = rep(table_ir) ^^ {
    _.toFastIndexedSeq
  }

  def ir_value_exprs: Parser[IndexedSeq[ir.IR]] = "(" ~> rep(ir_value_expr) <~ ")" ^^ {
    _.toFastIndexedSeq
  }

  def ir_value_exprs_opt: Parser[Option[IndexedSeq[ir.IR]]] =
    ir_value_exprs ^^ { xs => Some(xs) } |
      "None" ^^ { _ => None }

  def matrix_type_expr_opt: Parser[Option[MatrixType]] = matrix_type_expr ^^ {
    Some(_)
  } | "None" ^^ { _ => None }

  def type_exprs: Parser[Seq[Type]] = "(" ~> rep(type_expr) <~ ")"

  def type_exprs_opt: Parser[Option[Seq[Type]]] = type_exprs ^^ { ts => Some(ts) } | "None" ^^ { _ => None }

  def agg_signature: Parser[AggSignature] =
    "(" ~> ir_agg_op ~ type_exprs ~ type_exprs_opt ~ type_exprs <~ ")" ^^ { case op ~ ctorArgTypes ~ initOpArgTypes ~ seqOpArgTypes =>
      AggSignature(op, ctorArgTypes, initOpArgTypes, seqOpArgTypes)
    }

  def ir_named_value_exprs: Parser[Seq[(String, ir.IR)]] = rep(ir_named_value_expr)

  def ir_named_value_expr: Parser[(String, ir.IR)] =
    "(" ~> ir_identifier ~ ir_value_expr <~ ")" ^^ { case n ~ x => (n, x) }

  def ir_opt[T](p: Parser[T]): Parser[Option[T]] = p ^^ { Some(_) } | "None" ^^ { _ => None }

  def ir_value_expr: Parser[ir.IR] = "(" ~> ir_value_expr_1 <~ ")"

  def ir_value_expr_1: Parser[ir.IR] =
    "I32" ~> int32_literal ^^ { x => ir.I32(x) } |
      "I64" ~> int64_literal ^^ { x => ir.I64(x) } |
      "F32" ~> float32_literal ^^ { x => ir.F32(x) } |
      "F64" ~> float64_literal ^^ { x => ir.F64(x) } |
      "Str" ~> string_literal ^^ { x => ir.Str(x) } |
      "True" ^^ { x => ir.True() } |
      "False" ^^ { x => ir.False() } |
      "Void" ^^ { x => ir.Void() } |
      "Cast" ~> type_expr ~ ir_value_expr ^^ { case t ~ v => ir.Cast(v, t) } |
      "NA" ~> type_expr ^^ { t => ir.NA(t) } |
      "IsNA" ~> ir_value_expr ^^ { value => ir.IsNA(value) } |
      "If" ~> ir_value_expr ~ ir_value_expr ~ ir_value_expr ^^ { case cond ~ consq ~ altr => ir.If(cond, consq, altr) } |
      "Let" ~> ir_identifier ~ ir_value_expr ~ ir_value_expr ^^ { case name ~ value ~ body => ir.Let(name, value, body) } |
      "Ref" ~> type_expr ~ ir_identifier ^^ { case t ~ name => ir.Ref(name, t) } |
      "ApplyBinaryPrimOp" ~> ir_binary_op ~ ir_value_expr ~ ir_value_expr ^^ { case op ~ l ~ r => ir.ApplyBinaryPrimOp(op, l, r) } |
      "ApplyUnaryPrimOp" ~> ir_unary_op ~ ir_value_expr ^^ { case op ~ x => ir.ApplyUnaryPrimOp(op, x) } |
      "ApplyComparisonOp" ~> ir_comparison_op ~ ir_value_expr ~ ir_value_expr ^^ { case op ~ l ~ r => ir.ApplyComparisonOp(op, l, r) } |
      "MakeArray" ~> type_expr ~ ir_children ^^ { case t ~ args => ir.MakeArray(args, t.asInstanceOf[TArray]) } |
      "ArrayRef" ~> ir_value_expr ~ ir_value_expr ^^ { case a ~ i => ir.ArrayRef(a, i) } |
      "ArrayLen" ~> ir_value_expr ^^ { a => ir.ArrayLen(a) } |
      "ArrayRange" ~> ir_value_expr ~ ir_value_expr ~ ir_value_expr ^^ { case start ~ stop ~ step => ir.ArrayRange(start, stop, step) } |
      "ArraySort" ~> boolean_literal ~ ir_value_expr ~ ir_value_expr ^^ { case onKey ~ a ~ ascending => ir.ArraySort(a, ascending, onKey) } |
      "ToSet" ~> ir_value_expr ^^ { a => ir.ToSet(a) } |
      "ToDict" ~> ir_value_expr ^^ { a => ir.ToDict(a) } |
      "ToArray" ~> ir_value_expr ^^ { a => ir.ToArray(a) } |
      "LowerBoundOnOrderedCollection" ~> boolean_literal ~ ir_value_expr ~ ir_value_expr ^^ { case onKey ~ col ~ elem => ir.LowerBoundOnOrderedCollection(col, elem, onKey) } |
      "GroupByKey" ~> ir_value_expr ^^ { a => ir.GroupByKey(a) } |
      "ArrayMap" ~> ir_identifier ~ ir_value_expr ~ ir_value_expr ^^ { case name ~ a ~ body => ir.ArrayMap(a, name, body) } |
      "ArrayFilter" ~> ir_identifier ~ ir_value_expr ~ ir_value_expr ^^ { case name ~ a ~ body => ir.ArrayFilter(a, name, body) } |
      "ArrayFlatMap" ~> ir_identifier ~ ir_value_expr ~ ir_value_expr ^^ { case name ~ a ~ body => ir.ArrayFlatMap(a, name, body) } |
      "ArrayFold" ~> ir_identifier ~ ir_identifier ~ ir_value_expr ~ ir_value_expr ~ ir_value_expr ^^ { case accumName ~ valueName ~ a ~ zero ~ body => ir.ArrayFold(a, zero, accumName, valueName, body) } |
      "ArrayFor" ~> ir_identifier ~ ir_value_expr ~ ir_value_expr ^^ { case name ~ a ~ body => ir.ArrayFor(a, name, body) } |
      "ApplyAggOp" ~> agg_signature ~ ir_value_expr ~ ir_value_exprs ~ ir_value_exprs_opt ^^ { case aggSig ~ a ~ ctorArgs ~ initOpArgs => ir.ApplyAggOp(a, ctorArgs, initOpArgs, aggSig) } |
      "ApplyScanOp" ~> agg_signature ~ ir_value_expr ~ ir_value_exprs ~ ir_value_exprs_opt ^^ { case aggSig ~ a ~ ctorArgs ~ initOpArgs => ir.ApplyScanOp(a, ctorArgs, initOpArgs, aggSig) } |
      "InitOp" ~> agg_signature ~ ir_value_expr ~ ir_value_exprs ^^ { case aggSig ~ i ~ args => ir.InitOp(i, args, aggSig) } |
      "SeqOp" ~> agg_signature ~ ir_value_expr ~ ir_value_exprs ^^ { case aggSig ~ i ~ args => ir.SeqOp(i, args, aggSig) } |
      "Begin" ~> ir_children ^^ { xs => ir.Begin(xs) } |
      "MakeStruct" ~> ir_named_value_exprs ^^ { fields => ir.MakeStruct(fields) } |
      "SelectFields" ~> ir_identifiers ~ ir_value_expr ^^ { case fields ~ old => ir.SelectFields(old, fields) } |
      "InsertFields" ~> ir_value_expr ~ ir_named_value_exprs ^^ { case old ~ fields => ir.InsertFields(old, fields) } |
      "GetField" ~> ir_identifier ~ ir_value_expr ^^ { case name ~ o => ir.GetField(o, name) } |
      "MakeTuple" ~> ir_children ^^ { xs => ir.MakeTuple(xs) } |
      "GetTupleElement" ~> int32_literal ~ ir_value_expr ^^ { case idx ~ o => ir.GetTupleElement(o, idx) } |
      "StringSlice" ~> ir_value_expr ~ ir_value_expr ~ ir_value_expr ^^ { case s ~ start ~ end => ir.StringSlice(s, start, end) } |
      "StringLength" ~> ir_value_expr ^^ { s => ir.StringLength(s) } |
      "In" ~> type_expr ~ int32_literal ^^ { case t ~ i => ir.In(i, t) } |
      "Die" ~> type_expr ~ string_literal ^^ { case t ~ message => ir.Die(message, t) } |
      ("ApplyIR" | "ApplySpecial" | "Apply") ~> ir_identifier ~ ir_children ^^ { case function ~ args => ir.invoke(function, args: _*) } |
      "Uniroot" ~> ir_identifier ~ ir_value_expr ~ ir_value_expr ~ ir_value_expr ^^ { case name ~ f ~ min ~ max => ir.Uniroot(name, f, min, max) }

  def ir_value: Parser[(Type, Any)] = type_expr ~ string_literal ^^ { case t ~ vJSONStr =>
    val vJSON = JsonMethods.parse(vJSONStr)
    val v = JSONAnnotationImpex.importAnnotation(vJSON, t)
    (t, v)
  }

  def table_ir: Parser[ir.TableIR] = "(" ~> table_ir_1 <~ ")"

  def table_irs: Parser[IndexedSeq[ir.TableIR]] = "(" ~> rep(table_ir) <~ ")" ^^ {
    _.toFastIndexedSeq
  }

  def table_ir_1: Parser[ir.TableIR] =
  // FIXME TableImport
    "TableUnkey" ~> table_ir ^^ { t => ir.TableUnkey(t) } |
      "TableKeyBy" ~> ir_identifiers ~ int32_literal_opt ~ boolean_literal ~ table_ir ^^ { case key ~ nPartKeys ~ sort ~ child =>
        ir.TableKeyBy(child, key, nPartKeys, sort)
      } |
      "TableDistinct" ~> table_ir ^^ { t => ir.TableDistinct(t) } |
      "TableFilter" ~> table_ir ~ ir_value_expr ^^ { case child ~ pred => ir.TableFilter(child, pred) } |
      "TableRead" ~> string_literal ~ boolean_literal ~ ir_opt(table_type_expr) ^^ { case path ~ dropRows ~ typ =>
        TableIR.read(HailContext.get, path, dropRows, typ)
      } |
      "MatrixColsTable" ~> matrix_ir ^^ { child => ir.MatrixColsTable(child) } |
      "MatrixRowsTable" ~> matrix_ir ^^ { child => ir.MatrixRowsTable(child) } |
      "MatrixEntriesTable" ~> matrix_ir ^^ { child => ir.MatrixEntriesTable(child) } |
      "TableAggregateByKey" ~> table_ir ~ ir_value_expr ^^ { case child ~ expr => ir.TableAggregateByKey(child, expr) } |
      "TableJoin" ~> ir_identifier ~ table_ir ~ table_ir ^^ { case joinType ~ left ~ right => ir.TableJoin(left, right, joinType) } |
      "TableParallelize" ~> table_type_expr ~ ir_value ~ int32_literal_opt ^^ { case typ ~ ((rowsType, rows)) ~ nPartitions =>
        ir.TableParallelize(typ, rows.asInstanceOf[IndexedSeq[Row]], nPartitions)
      } |
      "TableMapRows" ~> string_literals_opt ~ int32_literal_opt ~ table_ir ~ ir_value_expr ^^ { case newKey ~ preservedKeyFields ~ child ~ newRow =>
        ir.TableMapRows(child, newRow, newKey, preservedKeyFields)
      } |
      "TableMapGlobals" ~> ir_value ~ table_ir ~ ir_value_expr ^^ { case ((t, v)) ~ child ~ newRow =>
        ir.TableMapGlobals(child, newRow,
          BroadcastRow(v.asInstanceOf[Row], t.asInstanceOf[TBaseStruct], HailContext.get.sc))
      } |
      "TableRange" ~> int32_literal ~ int32_literal ^^ { case n ~ nPartitions => ir.TableRange(n, nPartitions) } |
      "TableUnion" ~> table_ir_children ^^ { children => ir.TableUnion(children) } |
      "TableOrderBy" ~> ir_identifiers ~ table_ir ^^ { case identifiers ~ child =>
        ir.TableOrderBy(child, identifiers.map(i =>
          if (i.charAt(0) == 'A')
            SortField(i.substring(1), Ascending)
          else
            SortField(i.substring(1), Descending)))
      } |
      "TableExplode" ~> identifier ~ table_ir ^^ { case field ~ child => ir.TableExplode(child, field) }

  def matrix_ir: Parser[ir.MatrixIR] = "(" ~> matrix_ir_1 <~ ")"

  def matrix_ir_1: Parser[ir.MatrixIR] =
    "MatrixFilterCols" ~> matrix_ir ~ ir_value_expr ^^ { case child ~ pred => ir.MatrixFilterCols(child, pred) } |
      "MatrixFilterRows" ~> matrix_ir ~ ir_value_expr ^^ { case child ~ pred => ir.MatrixFilterRows(child, pred) } |
      "MatrixFilterEntries" ~> matrix_ir ~ ir_value_expr ^^ { case child ~ pred => ir.MatrixFilterEntries(child, pred) } |
      "MatrixMapCols" ~> string_literals_opt ~ matrix_ir ~ ir_value_expr ^^ { case newKey ~ child ~ newCol => ir.MatrixMapCols(child, newCol, newKey) } |
      "MatrixMapRows" ~> string_literals_opt ~ string_literals_opt ~ matrix_ir ~ ir_value_expr ^^ { case newKey ~ newPartitionKey ~ child ~ newCol =>
        val newKPK = ((newKey, newPartitionKey): @unchecked) match {
          case (Some(k), Some(pk)) => Some((k, pk))
          case (None, None) => None
        }
        ir.MatrixMapRows(child, newCol, newKPK)
      } |
      "MatrixMapEntries" ~> matrix_ir ~ ir_value_expr ^^ { case child ~ newEntries => ir.MatrixMapEntries(child, newEntries) } |
      "MatrixMapGlobals" ~> matrix_ir ~ ir_value_expr ~ ir_value ^^ { case child ~ newGlobals ~ ((t, v)) =>
        ir.MatrixMapGlobals(child, newGlobals,
          BroadcastRow(v.asInstanceOf[Row], t.asInstanceOf[TBaseStruct], HailContext.get.sc))
      } |
      "MatrixAggregateColsByKey" ~> matrix_ir ~ ir_value_expr ^^ { case child ~ agg => ir.MatrixAggregateColsByKey(child, agg) } |
      "MatrixAggregateRowsByKey" ~> matrix_ir ~ ir_value_expr ^^ { case child ~ agg => ir.MatrixAggregateRowsByKey(child, agg) } |
      "MatrixRange" ~> int32_literal ~ int32_literal ~ int32_literal_opt ~ boolean_literal ~ boolean_literal ^^ { case nRows ~ nCols ~ nPartitions ~ dropCols ~ dropRows =>
        MatrixIR.range(HailContext.get, nRows, nCols, nPartitions, dropCols, dropRows)
      } |
      "MatrixRead" ~> string_literal ~ boolean_literal ~ boolean_literal ~ matrix_type_expr_opt ^^ { case path ~ dropCols ~ dropRows ~ requestedType =>
        MatrixIR.read(HailContext.get, path, dropCols, dropRows, requestedType)
      } |
      "MatrixExplodeRows" ~> ir_identifiers ~ matrix_ir ^^ { case path ~ child => ir.MatrixExplodeRows(child, path)} |
      "MatrixChooseCols" ~> int32_literals ~ matrix_ir ^^ { case oldIndices ~ child => ir.MatrixChooseCols(child, oldIndices) } |
      "MatrixCollectColsByKey" ~> matrix_ir ^^ { child => ir.MatrixCollectColsByKey(child) }

  def parse_value_ir(s: String): IR = parse(ir_value_expr, s)

  def parse_table_ir(s: String): TableIR = parse(table_ir, s)

  def parse_matrix_ir(s: String): MatrixIR = parse(matrix_ir, s)
}
