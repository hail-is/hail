package is.hail.expr

import is.hail.HailContext
import is.hail.expr.ir.{AggSignature, BaseIR, IR, MatrixIR, TableIR}
import is.hail.expr.types._
import is.hail.expr.types.physical.PType
import is.hail.rvd.RVDType
import is.hail.table.{Ascending, Descending, SortField}
import is.hail.utils.StringEscapeUtils._
import is.hail.utils._
import is.hail.variant._
import org.json4s.jackson.{JsonMethods, Serialization}

import scala.util.parsing.combinator.JavaTokenParsers
import scala.util.parsing.input.{Position, Positional}
import scala.collection.JavaConverters._

case class Positioned[T](x: T) extends Positional

case class IRParserEnvironment(
  refMap: Map[String, Type] = Map.empty,
  irMap: Map[String, BaseIR] = Map.empty
) {
  def update(newRefMap: Map[String, Type] = Map.empty, newIRMap: Map[String, BaseIR] = Map.empty): IRParserEnvironment =
    copy(refMap = refMap ++ newRefMap, irMap = irMap ++ newIRMap)

  def withRefMap(newRefMap: Map[String, Type]): IRParserEnvironment = {
    assert(refMap.isEmpty)
    copy(refMap = newRefMap)
  }

  def +(t: (String, Type)): IRParserEnvironment = copy(refMap = refMap + t, irMap)
}

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
  def parse[T](parser: Parser[T], code: String): T = {
    parseAll(parser, code) match {
      case Success(result, _) => result
      case NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
    }
  }

  def parseType(code: String): Type = parse(type_expr, code)

  def parsePType(code: String): PType = parse(type_expr, code).physicalType

  def parseStructType(code: String): TStruct = parse(struct_expr, code)

  def parseRVDType(code: String): RVDType = parse(rvd_type_expr, code)

  def parseTableType(code: String): TableType = parse(table_type_expr, code)

  def parseMatrixType(code: String): MatrixType = parse(matrix_type_expr, code)

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

  def comma_delimited_doubles: Parser[Array[Double]] =
    repsep(floatingPointNumber, ",") ^^ (_.map(_.toDouble).toArray)

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

  def type_expr_opt: Parser[Option[Type]] = type_expr ^^ {
    Some(_)
  } | "None" ^^ { _ => None }

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

  def rvd_type_expr: Parser[RVDType] =
    ((("RVDType" | "OrderedRVDType") ~ "{" ~ "key" ~ ":" ~ "[") ~> key) ~ (trailing_keys <~ "]") ~
      (("," ~ "row" ~ ":") ~> struct_expr <~ "}") ^^ { case partitionKey ~ restKey ~ rowType =>
      RVDType(rowType, partitionKey ++ restKey)
    }

  def table_type_expr: Parser[TableType] =
    (("Table" ~ "{" ~ "global" ~ ":") ~> struct_expr) ~
      (("," ~ "key" ~ ":") ~> ("None" ^^ { _ => FastIndexedSeq() } | key ^^ { key => key.toFastIndexedSeq })) ~
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
      MatrixType.fromParts(globalType, colKey, colType, rowPartitionKey ++ rowRestKey, rowType, entryType)
    }

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
      "-inf" ^^ { _ => Double.NegativeInfinity } |
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

  def ir_untyped_comparison_op: Parser[String] =
    "(" ~> ir_identifier <~ ")" ^^ { x => x }

  def ir_agg_op: Parser[ir.AggOp] =
    ir_identifier ^^ { x => ir.AggOp.fromString(x) }

  def ir_children(env: IRParserEnvironment): Parser[IndexedSeq[ir.IR]] =
    rep(ir_value_expr(env)) ^^ (_.toFastIndexedSeq)

  def table_ir_children(env: IRParserEnvironment): Parser[IndexedSeq[ir.TableIR]] = rep(table_ir(env)) ^^ (_.toFastIndexedSeq)

  def matrix_ir_children(env: IRParserEnvironment): Parser[IndexedSeq[ir.MatrixIR]] = rep(matrix_ir(env)) ^^ (_.toFastIndexedSeq)

  def ir_value_exprs(env: IRParserEnvironment): Parser[IndexedSeq[ir.IR]] =
    "(" ~> rep(ir_value_expr(env)) <~ ")" ^^ (_.toFastIndexedSeq)

  def ir_value_exprs_opt(env: IRParserEnvironment): Parser[Option[IndexedSeq[ir.IR]]] =
    ir_value_exprs(env) ^^ { xs => Some(xs) } |
      "None" ^^ { _ => None }

  def matrix_type_expr_opt: Parser[Option[MatrixType]] = matrix_type_expr ^^ {
    Some(_)
  } | "None" ^^ { _ => None }

  def type_exprs: Parser[Seq[Type]] = "(" ~> rep(type_expr) <~ ")"

  def type_exprs_opt: Parser[Option[Seq[Type]]] = type_exprs ^^ { ts => Some(ts) } | "None" ^^ { _ => None }

  def agg_signature: Parser[AggSignature] =
    "(" ~> ir_agg_op ~ type_exprs ~ type_exprs_opt ~ type_exprs <~ ")" ^^ { case op ~ ctorArgTypes ~ initOpArgTypes ~ seqOpArgTypes =>
      AggSignature(op, ctorArgTypes.map(t => -t), initOpArgTypes.map(_.map(t => -t)), seqOpArgTypes.map(t => -t))
    }

  def ir_named_value_exprs(env: IRParserEnvironment): Parser[Seq[(String, ir.IR)]] =
    rep(ir_named_value_expr(env))

  def ir_named_value_expr(env: IRParserEnvironment): Parser[(String, ir.IR)] =
    "(" ~> ir_identifier ~ ir_value_expr(env) <~ ")" ^^ { case n ~ x => (n, x) }

  def ir_opt[T](p: Parser[T]): Parser[Option[T]] = p ^^ {
    Some(_)
  } | "None" ^^ { _ => None }

  def ir_value_expr(env: IRParserEnvironment): Parser[ir.IR] = "(" ~> ir_value_expr_1(env) <~ ")"

  def ir_value_expr_1(env: IRParserEnvironment): Parser[ir.IR] = {
    "I32" ~> int32_literal ^^ { x => ir.I32(x) } |
      "I64" ~> int64_literal ^^ { x => ir.I64(x) } |
      "F32" ~> float32_literal ^^ { x => ir.F32(x) } |
      "F64" ~> float64_literal ^^ { x => ir.F64(x) } |
      "Str" ~> string_literal ^^ { x => ir.Str(x) } |
      "True" ^^ { x => ir.True() } |
      "False" ^^ { x => ir.False() } |
      "Literal" ~> ir_value ^^ { value => ir.Literal.coerce(value._1, value._2) } |
      "Void" ^^ { x => ir.Void() } |
      "Cast" ~> type_expr ~ ir_value_expr(env) ^^ { case t ~ v => ir.Cast(v, t) } |
      "NA" ~> type_expr ^^ { t => ir.NA(t) } |
      "IsNA" ~> ir_value_expr(env) ^^ { value => ir.IsNA(value) } |
      "If" ~> ir_value_expr(env) ~ ir_value_expr(env) ~ ir_value_expr(env) ^^ { case cond ~ consq ~ altr => ir.If(cond, consq, altr) } |
      "Let" ~> ir_identifier ~ ir_value_expr(env) >> { case name ~ value =>
        ir_value_expr(env + (name -> value.typ)) ^^ { body => ir.Let(name, value, body) }} |
      "Ref" ~> ir_identifier ^^ { name => ir.Ref(name, env.refMap(name)) } |
      "ApplyBinaryPrimOp" ~> ir_binary_op ~ ir_value_expr(env) ~ ir_value_expr(env) ^^ { case op ~ l ~ r => ir.ApplyBinaryPrimOp(op, l, r) } |
      "ApplyUnaryPrimOp" ~> ir_unary_op ~ ir_value_expr(env) ^^ { case op ~ x => ir.ApplyUnaryPrimOp(op, x) } |
      "ApplyComparisonOp" ~> ir_untyped_comparison_op ~ ir_value_expr(env) ~ ir_value_expr(env) ^^ { case op ~ l ~ r =>
        ir.ApplyComparisonOp(ir.ComparisonOp.fromStringAndTypes(op, l.typ, r.typ), l, r) } |
      "MakeArray" ~> type_expr_opt ~ ir_children(env) ^^ { case t ~ args => ir.MakeArray.unify(args, t.map(_.asInstanceOf[TArray]).orNull) } |
      "ArrayRef" ~> ir_value_expr(env) ~ ir_value_expr(env) ^^ { case a ~ i => ir.ArrayRef(a, i) } |
      "ArrayLen" ~> ir_value_expr(env) ^^ { a => ir.ArrayLen(a) } |
      "ArrayRange" ~> ir_value_expr(env) ~ ir_value_expr(env) ~ ir_value_expr(env) ^^ { case start ~ stop ~ step => ir.ArrayRange(start, stop, step) } |
      "ArraySort" ~> boolean_literal ~ ir_value_expr(env) ~ ir_value_expr(env) ^^ { case onKey ~ a ~ ascending => ir.ArraySort(a, ascending, onKey) } |
      "ToSet" ~> ir_value_expr(env) ^^ { a => ir.ToSet(a) } |
      "ToDict" ~> ir_value_expr(env) ^^ { a => ir.ToDict(a) } |
      "ToArray" ~> ir_value_expr(env) ^^ { a => ir.ToArray(a) } |
      "LowerBoundOnOrderedCollection" ~> boolean_literal ~ ir_value_expr(env) ~ ir_value_expr(env) ^^ { case onKey ~ col ~ elem => ir.LowerBoundOnOrderedCollection(col, elem, onKey) } |
      "GroupByKey" ~> ir_value_expr(env) ^^ { a => ir.GroupByKey(a) } |
      "ArrayMap" ~> ir_identifier ~ ir_value_expr(env) >> { case name ~ a =>
        ir_value_expr(env + (name -> coerce[TArray](a.typ).elementType)) ^^ { body => ir.ArrayMap(a, name, body) }} |
      "ArrayFilter" ~> ir_identifier ~ ir_value_expr(env) >> { case name ~ a =>
        ir_value_expr(env + (name -> coerce[TArray](a.typ).elementType)) ^^ { body => ir.ArrayFilter(a, name, body) }} |
      "ArrayFlatMap" ~> ir_identifier ~ ir_value_expr(env) >> { case name ~ a =>
        ir_value_expr(env + (name -> coerce[TArray](a.typ).elementType)) ^^ { body => ir.ArrayFlatMap(a, name, body) }} |
      "ArrayFold" ~> ir_identifier ~ ir_identifier ~ ir_value_expr(env) ~ ir_value_expr(env) >> { case accumName ~ valueName ~ a ~ zero =>
        val eltType = coerce[TArray](a.typ).elementType
        ir_value_expr(env.update(Map(accumName -> zero.typ, valueName -> eltType))) ^^ { body => ir.ArrayFold(a, zero, accumName, valueName, body) }} |
      "ArrayScan" ~> ir_identifier ~ ir_identifier ~ ir_value_expr(env) ~ ir_value_expr(env) >> { case accumName ~ valueName ~ a ~ zero =>
        val eltType = coerce[TArray](a.typ).elementType
        ir_value_expr(env.update(Map(accumName -> zero.typ, valueName -> eltType))) ^^ { body => ir.ArrayScan(a, zero, accumName, valueName, body) }} |
      "ArrayFor" ~> ir_identifier ~ ir_value_expr(env) >> { case name ~ a =>
        ir_value_expr(env + (name -> coerce[TArray](a.typ).elementType)) ^^ { body => ir.ArrayFor(a, name, body) }} |
      "AggFilter" ~> ir_value_expr(env) ~ ir_value_expr(env) ^^ { case cond ~ aggIR => ir.AggFilter(cond, aggIR) } |
      "AggExplode" ~> ir_identifier ~ ir_value_expr(env) >> { case name ~ a =>
        ir_value_expr(env + (name -> coerce[TArray](a.typ).elementType)) ^^ { aggBody => ir.AggExplode(a, name, aggBody) }} |
      "AggGroupBy" ~> ir_value_expr(env) ~ ir_value_expr(env) ^^ { case key ~ aggIR => ir.AggGroupBy(key, aggIR) } |
      "ApplyAggOp" ~> agg_signature ~ ir_value_exprs(env) ~ ir_value_exprs(env) ~ ir_value_exprs_opt(env) ^^ { case aggSig ~ seqOpArgs ~ ctorArgs ~ initOpArgs => ir.ApplyAggOp(seqOpArgs, ctorArgs, initOpArgs, aggSig) } |
      "ApplyScanOp" ~> agg_signature ~ ir_value_exprs(env) ~ ir_value_exprs(env) ~ ir_value_exprs_opt(env) ^^ { case aggSig ~ seqOpArgs ~ ctorArgs ~ initOpArgs => ir.ApplyScanOp(seqOpArgs, ctorArgs, initOpArgs, aggSig) } |
      "InitOp" ~> agg_signature ~ ir_value_expr(env) ~ ir_value_exprs(env) ^^ { case aggSig ~ i ~ args => ir.InitOp(i, args, aggSig) } |
      "SeqOp" ~> agg_signature ~ ir_value_expr(env) ~ ir_value_exprs(env) ^^ { case aggSig ~ i ~ args => ir.SeqOp(i, args, aggSig) } |
      "Begin" ~> ir_children(env) ^^ { xs => ir.Begin(xs) } |
      "MakeStruct" ~> ir_named_value_exprs(env) ^^ { fields => ir.MakeStruct(fields) } |
      "SelectFields" ~> ir_identifiers ~ ir_value_expr(env) ^^ { case fields ~ old => ir.SelectFields(old, fields) } |
      "InsertFields" ~> ir_value_expr(env) ~ ir_named_value_exprs(env) ^^ { case old ~ fields => ir.InsertFields(old, fields) } |
      "GetField" ~> ir_identifier ~ ir_value_expr(env) ^^ { case name ~ o => ir.GetField(o, name) } |
      "MakeTuple" ~> ir_children(env) ^^ { xs => ir.MakeTuple(xs) } |
      "GetTupleElement" ~> int32_literal ~ ir_value_expr(env) ^^ { case idx ~ o => ir.GetTupleElement(o, idx) } |
      "StringSlice" ~> ir_value_expr(env) ~ ir_value_expr(env) ~ ir_value_expr(env) ^^ { case s ~ start ~ end => ir.StringSlice(s, start, end) } |
      "StringLength" ~> ir_value_expr(env) ^^ { s => ir.StringLength(s) } |
      "In" ~> type_expr ~ int32_literal ^^ { case t ~ i => ir.In(i, t) } |
      "Die" ~> type_expr ~ string_literal ^^ { case t ~ message => ir.Die(message, t) } |
      "ApplySeeded" ~> ir_identifier ~ int64_literal ~ ir_children(env) ^^ { case function ~ seed ~ args => ir.ApplySeeded(function, args, seed) } |
      ("ApplyIR" | "ApplySpecial" | "Apply") ~> ir_identifier ~ ir_children(env) ^^ { case function ~ args => ir.invoke(function, args: _*) } |
      "Uniroot" ~> ir_identifier ~ ir_value_expr(env) ~ ir_value_expr(env) ~ ir_value_expr(env) ^^ { case name ~ f ~ min ~ max => ir.Uniroot(name, f, min, max) } |
      "JavaIR" ~> ir_identifier ^^ { name => env.irMap(name).asInstanceOf[IR] }
  }

  def ir_value: Parser[(Type, Any)] = type_expr ~ string_literal ^^ { case t ~ vJSONStr =>
    val vJSON = JsonMethods.parse(vJSONStr)
    val v = JSONAnnotationImpex.importAnnotation(vJSON, t)
    (t, v)
  }

  def table_ir(env: IRParserEnvironment): Parser[ir.TableIR] = "(" ~> table_ir_1(env) <~ ")"

  def table_irs(env: IRParserEnvironment): Parser[IndexedSeq[ir.TableIR]] = "(" ~> rep(table_ir(env)) <~ ")" ^^ {
    _.toFastIndexedSeq
  }

  def table_ir_1(env: IRParserEnvironment): Parser[ir.TableIR] = {
    // FIXME TableImport
    "TableKeyBy" ~> ir_identifiers ~ boolean_literal ~ table_ir(env) ^^ { case key ~ isSorted ~ child =>
      ir.TableKeyBy(child, key, isSorted)
    } |
      "TableDistinct" ~> table_ir(env) ^^ { t => ir.TableDistinct(t) } |
      "TableFilter" ~> table_ir(env) >> { child =>
        ir_value_expr(env.withRefMap(child.typ.refMap)) ^^ { pred => ir.TableFilter(child, pred) }} |
      "TableRead" ~> string_literal ~ boolean_literal ~ ir_opt(table_type_expr) ^^ { case path ~ dropRows ~ typ =>
        TableIR.read(HailContext.get, path, dropRows, typ)
      } |
      "MatrixColsTable" ~> matrix_ir(env) ^^ { child => ir.MatrixColsTable(child) } |
      "MatrixRowsTable" ~> matrix_ir(env) ^^ { child => ir.MatrixRowsTable(child) } |
      "MatrixEntriesTable" ~> matrix_ir(env) ^^ { child => ir.MatrixEntriesTable(child) } |
      "TableAggregateByKey" ~> table_ir(env) >> { child =>
        ir_value_expr(env.withRefMap(child.typ.refMap)) ^^ { expr => ir.TableAggregateByKey(child, expr) }} |
      "TableKeyByAndAggregate" ~> int32_literal_opt ~ int32_literal ~ table_ir(env) >> { case nPartitions ~ bufferSize ~ child =>
        val newEnv = env.withRefMap(child.typ.refMap)
        ir_value_expr(newEnv) ~ ir_value_expr(newEnv) ^^ {
        case expr ~ newKey => ir.TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize) }} |
      "TableRepartition" ~> int32_literal ~ boolean_literal ~ table_ir(env) ^^ { case n ~ shuffle ~ child => ir.TableRepartition(child, n, shuffle) } |
      "TableHead" ~> int64_literal ~ table_ir(env) ^^ { case n ~ child => ir.TableHead(child, n) } |
      "TableJoin" ~> ir_identifier ~ int32_literal ~ table_ir(env) ~ table_ir(env) ^^ { case joinType ~ joinKey ~ left ~ right =>
        ir.TableJoin(left, right, joinType, joinKey) } |
      "TableLeftJoinRightDistinct" ~> ir_identifier ~ table_ir(env) ~ table_ir(env) ^^ { case root ~ left ~ right => ir.TableLeftJoinRightDistinct(left, right, root) } |
      "TableParallelize" ~> int32_literal_opt ~ ir_value_expr(env) ^^ { case nPartitions ~ rows =>
        ir.TableParallelize(rows, nPartitions)
      } |
      "TableMapRows" ~> table_ir(env) >> { child => ir_value_expr(env.withRefMap(child.typ.refMap)) ^^ { newRow => ir.TableMapRows(child, newRow) }} |
      "TableMapGlobals" ~> table_ir(env) >> { child => ir_value_expr(env.withRefMap(child.typ.refMap)) ^^ { newRow => ir.TableMapGlobals(child, newRow) }} |
      "TableRange" ~> int32_literal ~ int32_literal ^^ { case n ~ nPartitions => ir.TableRange(n, nPartitions) } |
      "TableUnion" ~> table_ir_children(env) ^^ { children => ir.TableUnion(children) } |
      "TableOrderBy" ~> ir_identifiers ~ table_ir(env) ^^ { case identifiers ~ child =>
        ir.TableOrderBy(child, identifiers.map(i =>
          if (i.charAt(0) == 'A')
            SortField(i.substring(1), Ascending)
          else
            SortField(i.substring(1), Descending)))
      } |
      "TableExplode" ~> ir_identifier ~ table_ir(env) ^^ { case field ~ child => ir.TableExplode(child, field) } |
      "LocalizeEntries" ~> string_literal ~ matrix_ir(env) ^^ { case field ~ child =>
        ir.LocalizeEntries(child, field)
      } |
      "TableRename" ~> string_literals ~ string_literals ~ string_literals ~ string_literals ~ table_ir(env) ^^ {
        case rowK ~ rowV ~ globalK ~ globalV ~ child => ir.TableRename(child, rowK.zip(rowV).toMap, globalK.zip(globalV).toMap)
      } |
      "JavaTable" ~> ir_identifier ^^ { ident => env.irMap(ident).asInstanceOf[TableIR] }
  }

  def matrix_ir(env: IRParserEnvironment): Parser[ir.MatrixIR] = "(" ~> matrix_ir_1(env) <~ ")"

  def matrix_ir_1(env: IRParserEnvironment): Parser[ir.MatrixIR] = {
    "MatrixFilterCols" ~> matrix_ir(env) >> { child => ir_value_expr(env.withRefMap(child.typ.refMap)) ^^ (pred => ir.MatrixFilterCols(child, pred)) } |
      "MatrixFilterRows" ~> matrix_ir(env) >> { child => ir_value_expr(env.withRefMap(child.typ.refMap)) ^^ (pred => ir.MatrixFilterRows(child, pred)) } |
      "MatrixFilterEntries" ~> matrix_ir(env) >> { child => ir_value_expr(env.withRefMap(child.typ.refMap)) ^^ (pred => ir.MatrixFilterEntries(child, pred)) } |
      "MatrixMapCols" ~> string_literals_opt ~ matrix_ir(env) >> { case newKey ~ child =>
        ir_value_expr(env.withRefMap(child.typ.refMap)) ^^ { newCol => ir.MatrixMapCols(child, newCol, newKey) }} |
      "MatrixKeyRowsBy" ~> ir_identifiers ~ boolean_literal ~ matrix_ir(env) ^^ { case key ~ isSorted ~ child => ir.MatrixKeyRowsBy(child, key, isSorted) } |
      "MatrixMapRows" ~> matrix_ir(env) >> { child => ir_value_expr(env.withRefMap(child.typ.refMap)) ^^ (newRow => ir.MatrixMapRows(child, newRow)) } |
      "MatrixMapEntries" ~> matrix_ir(env) >> { child => ir_value_expr(env.withRefMap(child.typ.refMap)) ^^ (newEntry => ir.MatrixMapEntries(child, newEntry)) } |
      "MatrixMapGlobals" ~> matrix_ir(env) >> { child => ir_value_expr(env.withRefMap(child.typ.refMap)) ^^ (newGlobals => ir.MatrixMapGlobals(child, newGlobals)) } |
      "MatrixAggregateColsByKey" ~> matrix_ir(env) >> { child =>
        val newEnv = env.withRefMap(child.typ.refMap)
        ir_value_expr(newEnv) ~ ir_value_expr(newEnv) ^^ {
          case entryExpr ~ colExpr => ir.MatrixAggregateColsByKey(child, entryExpr, colExpr) }} |
      "MatrixAggregateRowsByKey" ~> matrix_ir(env) >> { child =>
        val newEnv = env.withRefMap(child.typ.refMap)
        ir_value_expr(newEnv) ~ ir_value_expr(newEnv) ^^ {
          case entryExpr ~ rowExpr => ir.MatrixAggregateRowsByKey(child, entryExpr, rowExpr) }} |
      "MatrixRead" ~> matrix_type_expr_opt ~ boolean_literal ~ boolean_literal ~ string_literal ^^ {
        case typ ~ dropCols ~ dropRows ~ readerStr =>
          implicit val formats = ir.MatrixReader.formats
          val reader = Serialization.read[ir.MatrixReader](readerStr)
          ir.MatrixRead(typ.getOrElse(reader.fullType), dropCols, dropRows, reader)
      } |
      "TableToMatrixTable" ~> string_literals ~ string_literals ~ string_literals ~ string_literals ~ int32_literal_opt ~ table_ir(env) ^^ {
        case rowKey ~ colKey ~ rowFields ~ colFields ~ nPartitions ~ child =>
          ir.TableToMatrixTable(child, rowKey, colKey, rowFields, colFields, nPartitions)
      } |
      "MatrixAnnotateRowsTable" ~> string_literal ~ boolean_literal ~ matrix_ir(env) ~ table_ir(env) >> {
        case uid ~ hasKey ~ child ~ table => rep(ir_value_expr(env.withRefMap(child.typ.refMap))) ^^ {
          key =>
          val keyIRs = if (hasKey) Some(key.toFastIndexedSeq) else None
          ir.MatrixAnnotateRowsTable(child, table, uid, keyIRs)
      }} |
      "MatrixAnnotateColsTable" ~> string_literal ~ matrix_ir(env) ~ table_ir(env) ^^ {
        case root ~ child ~ table =>
          ir.MatrixAnnotateColsTable(child, table, root)
      } |
      "MatrixExplodeRows" ~> ir_identifiers ~ matrix_ir(env) ^^ { case path ~ child => ir.MatrixExplodeRows(child, path) } |
      "MatrixExplodeCols" ~> ir_identifiers ~ matrix_ir(env) ^^ { case path ~ child => ir.MatrixExplodeCols(child, path) } |
      "MatrixChooseCols" ~> int32_literals ~ matrix_ir(env) ^^ { case oldIndices ~ child => ir.MatrixChooseCols(child, oldIndices) } |
      "MatrixCollectColsByKey" ~> matrix_ir(env) ^^ { child => ir.MatrixCollectColsByKey(child) } |
      "MatrixUnionRows" ~> matrix_ir_children(env) ^^ { children => ir.MatrixUnionRows(children) } |
      "UnlocalizeEntries" ~> string_literal ~ table_ir(env) ~ table_ir(env) ^^ {
        case entryField ~ rowsEntries ~ cols => ir.UnlocalizeEntries(rowsEntries, cols, entryField)
      } |
      "JavaMatrix" ~> ir_identifier ^^ { ident => env.irMap(ident).asInstanceOf[MatrixIR] }
  }

  def parse_value_ir(s: String): IR = parse_value_ir(s, IRParserEnvironment())
  def parse_value_ir(s: String, refMap: java.util.HashMap[String, Type]): IR = parse_value_ir(s, IRParserEnvironment(refMap = refMap.asScala.toMap))
  def parse_value_ir(s: String, env: IRParserEnvironment): IR = parse(ir_value_expr(env), s)

  def parse_named_value_irs(s: String): Array[(String, IR)] = parse_named_value_irs(s, IRParserEnvironment())
  def parse_named_value_irs(s: String, env: IRParserEnvironment): Array[(String, IR)] = parse(ir_named_value_exprs(env), s).toArray

  def parse_table_ir(s: String): TableIR = parse_table_ir(s, IRParserEnvironment())
  def parse_table_ir(s: String, env: IRParserEnvironment): TableIR = parse(table_ir(env), s)

  def parse_matrix_ir(s: String): MatrixIR = parse(matrix_ir(IRParserEnvironment()), s)
  def parse_matrix_ir(s: String, env: IRParserEnvironment): MatrixIR = parse(matrix_ir(env), s)
}
