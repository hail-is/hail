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
import org.json4s.jackson.{JsonMethods, Serialization}

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
      new OrderedRVDType(partitionKey ++ restKey, rowType)
    }

  def table_type_expr: Parser[TableType] =
    (("Table" ~ "{" ~ "global" ~ ":") ~> struct_expr) ~
      (("," ~ "key" ~ ":") ~> ("None" ^^ { _ => None } | key ^^ { key => Some(key.toFastIndexedSeq) })) ~
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
    ir_keyed_agg_op | ir_identifier ^^ { x => ir.AggOp.fromString(x) }

  def ir_keyed_agg_op: Parser[ir.Keyed] =
    "Keyed(" ~> ir_agg_op <~ ")" ^^ { aggOp => ir.Keyed(aggOp) }

  def ir_children(ref_map: Map[String, Type] = Map.empty): Parser[IndexedSeq[ir.IR]] =
    rep(ir_value_expr(ref_map)) ^^ { _.toFastIndexedSeq }

  def table_ir_children: Parser[IndexedSeq[ir.TableIR]] = rep(table_ir) ^^ {
    _.toFastIndexedSeq
  }

  def matrix_ir_children: Parser[IndexedSeq[ir.MatrixIR]] = rep(matrix_ir) ^^ {
    _.toFastIndexedSeq
  }

  def ir_value_exprs(ref_map: Map[String, Type] = Map.empty): Parser[IndexedSeq[ir.IR]] =
    "(" ~> rep(ir_value_expr(ref_map)) <~ ")" ^^ { _.toFastIndexedSeq }

  def ir_value_exprs_opt(ref_map: Map[String, Type] = Map.empty): Parser[Option[IndexedSeq[ir.IR]]] =
    ir_value_exprs(ref_map) ^^ { xs => Some(xs) } |
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

  def ir_named_value_exprs(ref_map: Map[String, Type] = Map.empty): Parser[Seq[(String, ir.IR)]] = rep(ir_named_value_expr(ref_map))

  def ir_named_value_expr(ref_map: Map[String, Type] = Map.empty): Parser[(String, ir.IR)] =
    "(" ~> ir_identifier ~ ir_value_expr(ref_map) <~ ")" ^^ { case n ~ x => (n, x) }

  def ir_opt[T](p: Parser[T]): Parser[Option[T]] = p ^^ { Some(_) } | "None" ^^ { _ => None }

  def ir_value_expr(ref_map: Map[String, Type] = Map.empty): Parser[ir.IR] = "(" ~> ir_value_expr_1(ref_map) <~ ")"

  def ir_value_expr_1(ref_map: Map[String, Type] = Map.empty): Parser[ir.IR] = {
    def expr_with_map: Parser[ir.IR] = ir_value_expr(ref_map)
    "I32" ~> int32_literal ^^ { x => ir.I32(x) } |
      "I64" ~> int64_literal ^^ { x => ir.I64(x) } |
      "F32" ~> float32_literal ^^ { x => ir.F32(x) } |
      "F64" ~> float64_literal ^^ { x => ir.F64(x) } |
      "Str" ~> string_literal ^^ { x => ir.Str(x) } |
      "True" ^^ { x => ir.True() } |
      "False" ^^ { x => ir.False() } |
      "Literal" ~> ir_value ~ string_literal ^^ { case (value ~ id) => ir.Literal(value._1, value._2, id)} |
      "Void" ^^ { x => ir.Void() } |
      "Cast" ~> type_expr ~ expr_with_map ^^ { case t ~ v => ir.Cast(v, t) } |
      "NA" ~> type_expr ^^ { t => ir.NA(t) } |
      "IsNA" ~> expr_with_map ^^ { value => ir.IsNA(value) } |
      "If" ~> expr_with_map ~ expr_with_map ~ expr_with_map ^^ { case cond ~ consq ~ altr => ir.If(cond, consq, altr) } |
      "Let" ~> ir_identifier ~ expr_with_map ~ expr_with_map ^^ { case name ~ value ~ body => ir.Let(name, value, body) } |
      "Ref" ~> type_expr ~ ir_identifier ^^ { case t ~ name => ir.Ref(name, t) } |
      "Ref" ~> ir_identifier ^^ { name => ir.Ref(name, ref_map(name)) } |
      "ApplyBinaryPrimOp" ~> ir_binary_op ~ expr_with_map ~ expr_with_map ^^ { case op ~ l ~ r => ir.ApplyBinaryPrimOp(op, l, r) } |
      "ApplyUnaryPrimOp" ~> ir_unary_op ~ expr_with_map ^^ { case op ~ x => ir.ApplyUnaryPrimOp(op, x) } |
      "ApplyComparisonOp" ~> ir_comparison_op ~ expr_with_map ~ expr_with_map ^^ { case op ~ l ~ r => ir.ApplyComparisonOp(op, l, r) } |
      "ApplyComparisonOp" ~> ir_untyped_comparison_op ~ expr_with_map ~ expr_with_map ^^ { case op ~ l ~ r => ir.ApplyComparisonOp(ir.ComparisonOp.fromStringAndTypes(op, l.typ, r.typ), l, r) } |
      "MakeArray" ~> type_expr ~ ir_children(ref_map) ^^ { case t ~ args => ir.MakeArray(args, t.asInstanceOf[TArray]) } |
      "ArrayRef" ~> expr_with_map  ~ expr_with_map ^^ { case a ~ i => ir.ArrayRef(a, i) } |
      "ArrayLen" ~> expr_with_map ^^ { a => ir.ArrayLen(a) } |
      "ArrayRange" ~> expr_with_map  ~ expr_with_map ~ expr_with_map  ^^ { case start ~ stop ~ step => ir.ArrayRange(start, stop, step) } |
      "ArraySort" ~> boolean_literal ~ expr_with_map ~ expr_with_map ^^ { case onKey ~ a ~ ascending => ir.ArraySort(a, ascending, onKey) } |
      "ToSet" ~> expr_with_map ^^ { a => ir.ToSet(a) } |
      "ToDict" ~> expr_with_map ^^ { a => ir.ToDict(a) } |
      "ToArray" ~> expr_with_map ^^ { a => ir.ToArray(a) } |
      "LowerBoundOnOrderedCollection" ~> boolean_literal ~ expr_with_map ~ expr_with_map ^^ { case onKey ~ col ~ elem => ir.LowerBoundOnOrderedCollection(col, elem, onKey) } |
      "GroupByKey" ~> expr_with_map ^^ { a => ir.GroupByKey(a) } |
      "ArrayMap" ~> ir_identifier ~ expr_with_map ~ expr_with_map ^^ { case name ~ a ~ body => ir.ArrayMap(a, name, body) } |
      "ArrayFilter" ~> ir_identifier ~ expr_with_map ~ expr_with_map ^^ { case name ~ a ~ body => ir.ArrayFilter(a, name, body) } |
      "ArrayFlatMap" ~> ir_identifier ~ expr_with_map ~ expr_with_map ^^ { case name ~ a ~ body => ir.ArrayFlatMap(a, name, body) } |
      "ArrayFold" ~> ir_identifier ~ ir_identifier ~ expr_with_map ~ expr_with_map ~ expr_with_map ^^ { case accumName ~ valueName ~ a ~ zero ~ body => ir.ArrayFold(a, zero, accumName, valueName, body) } |
      "ArrayFor" ~> ir_identifier ~ expr_with_map ~ expr_with_map ^^ { case name ~ a ~ body => ir.ArrayFor(a, name, body) } |
      "ApplyAggOp" ~> agg_signature ~ expr_with_map ~ ir_value_exprs(ref_map) ~ ir_value_exprs_opt(ref_map) ^^ { case aggSig ~ a ~ ctorArgs ~ initOpArgs => ir.ApplyAggOp(a, ctorArgs, initOpArgs, aggSig) } |
      "ApplyScanOp" ~> agg_signature ~ expr_with_map ~ ir_value_exprs(ref_map) ~ ir_value_exprs_opt(ref_map) ^^ { case aggSig ~ a ~ ctorArgs ~ initOpArgs => ir.ApplyScanOp(a, ctorArgs, initOpArgs, aggSig) } |
      "InitOp" ~> agg_signature ~ expr_with_map ~ ir_value_exprs(ref_map) ^^ { case aggSig ~ i ~ args => ir.InitOp(i, args, aggSig) } |
      "SeqOp" ~> agg_signature ~ expr_with_map ~ ir_value_exprs(ref_map) ^^ { case aggSig ~ i ~ args => ir.SeqOp(i, args, aggSig) } |
      "Begin" ~> ir_children(ref_map) ^^ { xs => ir.Begin(xs) } |
      "MakeStruct" ~> ir_named_value_exprs(ref_map) ^^ { fields => ir.MakeStruct(fields) } |
      "SelectFields" ~> ir_identifiers ~ expr_with_map ^^ { case fields ~ old => ir.SelectFields(old, fields) } |
      "InsertFields" ~> expr_with_map ~ ir_named_value_exprs(ref_map) ^^ { case old ~ fields => ir.InsertFields(old, fields) } |
      "GetField" ~> ir_identifier ~ expr_with_map ^^ { case name ~ o => ir.GetField(o, name) } |
      "MakeTuple" ~> ir_children(ref_map) ^^ { xs => ir.MakeTuple(xs) } |
      "GetTupleElement" ~> int32_literal ~ expr_with_map ^^ { case idx ~ o => ir.GetTupleElement(o, idx) } |
      "StringSlice" ~> expr_with_map ~ expr_with_map ~ expr_with_map ^^ { case s ~ start ~ end => ir.StringSlice(s, start, end) } |
      "StringLength" ~> expr_with_map ^^ { s => ir.StringLength(s) } |
      "In" ~> type_expr ~ int32_literal ^^ { case t ~ i => ir.In(i, t) } |
      "Die" ~> type_expr ~ string_literal ^^ { case t ~ message => ir.Die(message, t) } |
      "ApplySeeded" ~> ir_identifier ~ int64_literal ~ ir_children(ref_map) ^^ { case function ~ seed ~ args => ir.ApplySeeded(function, args, seed) } |
      ("ApplyIR" | "ApplySpecial" | "Apply") ~> ir_identifier ~ ir_children(ref_map) ^^ { case function ~ args => ir.invoke(function, args: _*) } |
      "Uniroot" ~> ir_identifier ~ expr_with_map ~ expr_with_map ~ expr_with_map ^^ { case name ~ f ~ min ~ max => ir.Uniroot(name, f, min, max) }
  }


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
      "TableKeyBy" ~> ir_identifiers ~ boolean_literal ~ table_ir ^^ { case key ~ isSorted ~ child =>
        ir.TableKeyBy(child, key, isSorted)
      } |
      "TableDistinct" ~> table_ir ^^ { t => ir.TableDistinct(t) } |
      "TableFilter" ~> table_ir ~ ir_value_expr() ^^ { case child ~ pred => ir.TableFilter(child, pred) } |
      "TableRead" ~> string_literal ~ boolean_literal ~ ir_opt(table_type_expr) ^^ { case path ~ dropRows ~ typ =>
        TableIR.read(HailContext.get, path, dropRows, typ)
      } |
      "MatrixColsTable" ~> matrix_ir ^^ { child => ir.MatrixColsTable(child) } |
      "MatrixRowsTable" ~> matrix_ir ^^ { child => ir.MatrixRowsTable(child) } |
      "MatrixEntriesTable" ~> matrix_ir ^^ { child => ir.MatrixEntriesTable(child) } |
      "TableAggregateByKey" ~> table_ir ~ ir_value_expr() ^^ { case child ~ expr => ir.TableAggregateByKey(child, expr) } |
      "TableKeyByAndAggregate" ~> int32_literal_opt ~ int32_literal ~ table_ir ~ ir_value_expr() ~ ir_value_expr() ^^ {
        case nPartitions ~ bufferSize ~ child ~ expr ~ newKey => ir.TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize) } |
      "TableRepartition" ~> int32_literal ~ boolean_literal ~ table_ir ^^ { case n ~ shuffle ~ child => ir.TableRepartition(child, n, shuffle) } |
      "TableHead" ~> int64_literal ~ table_ir ^^ { case n ~ child => ir.TableHead(child, n) } |
      "TableJoin" ~> ir_identifier ~ table_ir ~ table_ir ^^ { case joinType ~ left ~ right => ir.TableJoin(left, right, joinType) } |
      "TableLeftJoinRightDistinct" ~> ir_identifier ~ table_ir ~ table_ir ^^ { case root ~ left ~ right => ir.TableLeftJoinRightDistinct(left, right, root) } |
      "TableParallelize" ~> table_type_expr ~ ir_value ~ int32_literal_opt ^^ { case typ ~ ((rowsType, rows)) ~ nPartitions =>
        ir.TableParallelize(typ, rows.asInstanceOf[IndexedSeq[Row]], nPartitions)
      } |
      "TableMapRows" ~> string_literals_opt ~ int32_literal_opt ~ table_ir ~ ir_value_expr() ^^ { case newKey ~ preservedKeyFields ~ child ~ newRow =>
        ir.TableMapRows(child, newRow, newKey, preservedKeyFields)
      } |
      "TableMapGlobals" ~> table_ir ~ ir_value_expr() ^^ { case child ~ newRow =>
        ir.TableMapGlobals(child, newRow)
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
      "TableExplode" ~> identifier ~ table_ir ^^ { case field ~ child => ir.TableExplode(child, field) } |
      "LocalizeEntries" ~> string_literal ~ matrix_ir ^^ { case field ~ child =>
        ir.LocalizeEntries(child, field)
      }

  def matrix_ir: Parser[ir.MatrixIR] = "(" ~> matrix_ir_1 <~ ")"

  def matrix_ir_1: Parser[ir.MatrixIR] =
    "MatrixFilterCols" ~> matrix_ir ~ ir_value_expr() ^^ { case child ~ pred => ir.MatrixFilterCols(child, pred) } |
      "MatrixFilterRows" ~> matrix_ir ~ ir_value_expr() ^^ { case child ~ pred => ir.MatrixFilterRows(child, pred) } |
      "MatrixFilterEntries" ~> matrix_ir ~ ir_value_expr() ^^ { case child ~ pred => ir.MatrixFilterEntries(child, pred) } |
      "MatrixMapCols" ~> string_literals_opt ~ matrix_ir ~ ir_value_expr() ^^ { case newKey ~ child ~ newCol => ir.MatrixMapCols(child, newCol, newKey) } |
      "MatrixMapRows" ~> string_literals_opt ~ string_literals_opt ~ matrix_ir ~ ir_value_expr() ^^ { case newKey ~ newPartitionKey ~ child ~ newRow =>
        val newKPK = ((newKey, newPartitionKey): @unchecked) match {
          case (Some(k), Some(pk)) => Some((k, pk))
          case (None, None) => None
        }
        ir.MatrixMapRows(child, newRow, newKPK)
      } |
      "MatrixMapEntries" ~> matrix_ir ~ ir_value_expr() ^^ { case child ~ newEntries => ir.MatrixMapEntries(child, newEntries) } |
      "MatrixMapGlobals" ~> matrix_ir ~ ir_value_expr() ^^ { case child ~ newGlobals =>
        ir.MatrixMapGlobals(child, newGlobals)
      } |
      "MatrixAggregateColsByKey" ~> matrix_ir ~ ir_value_expr() ^^ { case child ~ agg => ir.MatrixAggregateColsByKey(child, agg) } |
      "MatrixAggregateRowsByKey" ~> matrix_ir ~ ir_value_expr() ^^ { case child ~ agg => ir.MatrixAggregateRowsByKey(child, agg) } |
      "MatrixRead" ~> matrix_type_expr_opt ~ boolean_literal ~ boolean_literal ~ string_literal ^^ {
        case typ ~ dropCols ~ dropRows ~ readerStr =>
          implicit val formats = ir.MatrixReader.formats
          val reader = Serialization.read[ir.MatrixReader](readerStr)
          ir.MatrixRead(typ.getOrElse(reader.fullType), dropCols, dropRows, reader)
      } |
      "TableToMatrixTable" ~> string_literals ~ string_literals ~ string_literals ~ string_literals ~ string_literals ~ int32_literal_opt ~ table_ir ^^ {
        case rowKey ~ colKey ~ rowFields ~ colFields ~ partitionKey ~ nPartitions ~ child =>
          ir.TableToMatrixTable(child, rowKey, colKey, rowFields, colFields, partitionKey, nPartitions)
      } |
      "MatrixAnnotateRowsTable" ~> string_literal ~ boolean_literal ~ matrix_ir ~ table_ir ~ rep(ir_value_expr()) ^^ {
        case uid ~ hasKey ~ child ~ table ~ key =>
          val keyIRs = if (hasKey) Some(key.toFastIndexedSeq) else None
          ir.MatrixAnnotateRowsTable(child, table, uid, keyIRs)
      } |
      "MatrixAnnotateColsTable" ~> string_literal ~ matrix_ir ~ table_ir ^^ {
        case root ~ child ~ table =>
          ir.MatrixAnnotateColsTable(child, table, root)
      } |
      "MatrixExplodeRows" ~> ir_identifiers ~ matrix_ir ^^ { case path ~ child => ir.MatrixExplodeRows(child, path)} |
      "MatrixExplodeCols" ~> ir_identifiers ~ matrix_ir ^^ { case path ~ child => ir.MatrixExplodeCols(child, path)} |
      "MatrixChooseCols" ~> int32_literals ~ matrix_ir ^^ { case oldIndices ~ child => ir.MatrixChooseCols(child, oldIndices) } |
      "MatrixCollectColsByKey" ~> matrix_ir ^^ { child => ir.MatrixCollectColsByKey(child) } |
      "MatrixUnionRows" ~> matrix_ir_children ^^ { children => ir.MatrixUnionRows(children) } |
      "UnlocalizeEntries" ~> string_literal ~ table_ir ~ table_ir ^^ {
        case entryField ~ rowsEntries ~ cols => ir.UnlocalizeEntries(rowsEntries, cols, entryField)
      }

  def parse_value_ir(s: String, ref_map: Map[String, Type]): IR = parse(ir_value_expr(ref_map), s)

  def parse_value_ir(s: String): IR = parse_value_ir(s, Map.empty)

  def parse_named_value_irs(s: String, ref_map: Map[String, Type] = Map.empty): Array[(String, IR)] =
    parse(ir_named_value_exprs(ref_map), s).toArray

  def parse_table_ir(s: String): TableIR = parse(table_ir, s)

  def parse_matrix_ir(s: String): MatrixIR = parse(matrix_ir, s)
}
