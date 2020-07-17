package is.hail.expr.ir

import is.hail.HailContext
import is.hail.expr.ir.agg._
import is.hail.expr.ir.functions.RelationalFunctions
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.types.encoded._
import is.hail.types.{MatrixType, TableType}
import is.hail.expr.{JSONAnnotationImpex, Nat, ParserUtils}
import is.hail.io.{AbstractTypedCodecSpec, BufferSpec}
import is.hail.rvd.{AbstractRVDSpec, RVDType}
import is.hail.utils.StringEscapeUtils._
import is.hail.utils._
import is.hail.variant.ReferenceGenome
import org.apache.spark.sql.Row
import org.json4s.{Formats, JObject}
import org.json4s.jackson.{JsonMethods, Serialization}

import scala.collection.JavaConverters._
import scala.reflect.ClassTag
import scala.util.parsing.combinator.JavaTokenParsers
import scala.util.parsing.input.Positional

abstract class Token extends Positional {
  def value: Any

  def getName: String
}

final case class IdentifierToken(value: String) extends Token {
  def getName: String = "identifier"
}

final case class StringToken(value: String) extends Token {
  def getName: String = "string"
}

final case class IntegerToken(value: Long) extends Token {
  def getName: String = "integer"
}

final case class FloatToken(value: Double) extends Token {
  def getName: String = "float"
}

final case class PunctuationToken(value: String) extends Token {
  def getName: String = "punctuation"
}

object IRLexer extends JavaTokenParsers {
  val token: Parser[Token] =
    identifier ^^ { id => IdentifierToken(id) } |
      float64_literal ^^ { d => FloatToken(d) } |
      int64_literal ^^ { l => IntegerToken(l) } |
      string_literal ^^ { s => StringToken(s) } |
      "[()\\[\\]{}<>,:+@=]".r ^^ { p => PunctuationToken(p) }

  val lexer: Parser[Array[Token]] = rep(positioned(token)) ^^ { l => l.toArray }

  def quotedLiteral(delim: Char, what: String): Parser[String] =
    new Parser[String] {
      def apply(in: Input): ParseResult[String] = {
        var r = in

        val source = in.source
        val offset = in.offset
        val start = handleWhiteSpace(source, offset)
        r = r.drop(start - offset)

        if (r.atEnd || r.first != delim)
          return Failure(s"consumed $what", r)
        r = r.rest

        val sb = new StringBuilder()

        val escapeChars = "\\bfnrtu'\"`".toSet
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

  override def stringLiteral: Parser[String] =
    quotedLiteral('"', "string literal") | quotedLiteral('\'', "string literal")

  def backtickLiteral: Parser[String] = quotedLiteral('`', "backtick identifier")

  def identifier = backtickLiteral | ident

  def string_literal: Parser[String] = stringLiteral

  def int64_literal: Parser[Long] = wholeNumber.map(_.toLong)

  def float64_literal: Parser[Double] =
      "-inf" ^^ { _ => Double.NegativeInfinity } | // inf, neginf, and nan are parsed as identifiers
      """[+-]?\d+(\.\d+)?[eE][+-]?\d+""".r ^^ { _.toDouble } |
      """[+-]?\d*\.\d+""".r ^^ { _.toDouble }

  def parse(code: String): Array[Token] = {
    parseAll(lexer, code) match {
      case Success(result, _) => result
      case NoSuccess(msg, next) => ParserUtils.error(next.pos, msg)
    }
  }
}

object TypeParserEnvironment {
  def empty: TypeParserEnvironment = TypeParserEnvironment(Map.empty)

  // FIXME: This will go away when references are no longer kept on global object
  def default: TypeParserEnvironment = TypeParserEnvironment(ReferenceGenome.references)
}

case class TypeParserEnvironment(
  rgMap: Map[String, ReferenceGenome]
) {
  def getReferenceGenome(name: String): ReferenceGenome = rgMap(name)

}

case class IRParserEnvironment(
  ctx: ExecuteContext,
  refMap: Map[String, Type] = Map.empty,
  irMap: Map[String, BaseIR] = Map.empty,
  typEnv: TypeParserEnvironment = TypeParserEnvironment.default
) {
  def update(newRefMap: Map[String, Type] = Map.empty, newIRMap: Map[String, BaseIR] = Map.empty): IRParserEnvironment =
    copy(refMap = refMap ++ newRefMap, irMap = irMap ++ newIRMap)

  def withRefMap(newRefMap: Map[String, Type]): IRParserEnvironment = {
    assert(refMap.isEmpty || newRefMap.isEmpty)
    copy(refMap = newRefMap)
  }

  def +(t: (String, Type)): IRParserEnvironment = copy(refMap = refMap + t)
  def ++(ts: Array[(String, Type)]): IRParserEnvironment = copy(refMap = refMap ++ ts)
}

object IRParser {
  def error(t: Token, msg: String): Nothing = ParserUtils.error(t.pos, msg)

  def deserialize[T](str: String)(implicit formats: Formats, mf: Manifest[T]): T = {
    Serialization.read[T](str)
  }

  def consumeToken(it: TokenIterator): Token = {
    if (!it.hasNext)
      fatal("No more tokens to consume.")
    it.next()
  }

  def punctuation(it: TokenIterator, symbol: String): String = {
    consumeToken(it) match {
      case x: PunctuationToken if x.value == symbol => x.value
      case x: Token => error(x, s"Expected punctuation '$symbol' but found ${ x.getName } '${ x.value }'.")
    }
  }

  def identifier(it: TokenIterator): String = {
    consumeToken(it) match {
      case x: IdentifierToken => x.value
      case x: Token => error(x, s"Expected identifier but found ${ x.getName } '${ x.value }'.")
    }
  }

  def identifier(it: TokenIterator, expectedId: String): String = {
    consumeToken(it) match {
      case x: IdentifierToken if x.value == expectedId => x.value
      case x: Token => error(x, s"Expected identifier '$expectedId' but found ${ x.getName } '${ x.value }'.")
    }
  }

  def identifiers(it: TokenIterator): Array[String] =
    base_seq_parser(identifier)(it)

  def boolean_literal(it: TokenIterator): Boolean = {
    consumeToken(it) match {
      case IdentifierToken("True") => true
      case IdentifierToken("False") => false
      case x: Token => error(x, s"Expected boolean but found ${ x.getName } '${ x.value }'.")
    }
  }

  def int32_literal(it: TokenIterator): Int = {
    consumeToken(it) match {
      case x: IntegerToken =>
        if (x.value >= Int.MinValue && x.value <= Int.MaxValue)
          x.value.toInt
        else
          error(x, s"Found integer '${ x.value }' that is outside the numeric range for int32.")
      case x: Token => error(x, s"Expected integer but found ${ x.getName } '${ x.value }'.")
    }
  }

  def int64_literal(it: TokenIterator): Long = {
    consumeToken(it) match {
      case x: IntegerToken => x.value
      case x: Token => error(x, s"Expected integer but found ${ x.getName } '${ x.value }'.")
    }
  }

  def float32_literal(it: TokenIterator): Float = {
    consumeToken(it) match {
      case x: FloatToken =>
        if (x.value >= Float.MinValue && x.value <= Float.MaxValue)
          x.value.toFloat
        else
          error(x, s"Found float '${ x.value }' that is outside the numeric range for float32.")
      case x: IntegerToken => x.value.toFloat
      case x: IdentifierToken => x.value match {
        case "nan" => Float.NaN
        case "inf" => Float.PositiveInfinity
        case "neginf" => Float.NegativeInfinity
        case _ => error(x, s"Expected float but found ${ x.getName } '${ x.value }'.")
      }
      case x: Token => error(x, s"Expected float but found ${ x.getName } '${ x.value }'.")
    }
  }

  def float64_literal(it: TokenIterator): Double = {
    consumeToken(it) match {
      case x: FloatToken => x.value
      case x: IntegerToken => x.value.toDouble
      case x: IdentifierToken => x.value match {
        case "nan" => Double.NaN
        case "inf" => Double.PositiveInfinity
        case "neginf" => Double.NegativeInfinity
        case _ => error(x, s"Expected float but found ${ x.getName } '${ x.value }'.")
      }
      case x: Token => error(x, s"Expected float but found ${ x.getName } '${ x.value }'.")
    }
  }

  def string_literal(it: TokenIterator): String = {
    consumeToken(it) match {
      case x: StringToken => x.value
      case x: Token => error(x, s"Expected string but found ${ x.getName } '${ x.value }'.")
    }
  }

  def literals[T](literalIdentifier: TokenIterator => T)(it: TokenIterator)(implicit tct: ClassTag[T]): Array[T] =
    base_seq_parser(literalIdentifier)(it)

  def string_literals: TokenIterator => Array[String] = literals(string_literal)
  def int32_literals: TokenIterator => Array[Int] = literals(int32_literal)
  def int64_literals: TokenIterator => Array[Long] = literals(int64_literal)

  def opt[T](it: TokenIterator, f: (TokenIterator) => T)(implicit tct: ClassTag[T]): Option[T] = {
    it.head match {
      case x: IdentifierToken if x.value == "None" =>
        consumeToken(it)
        None
      case _ =>
        Some(f(it))
    }
  }

  def repsepUntil[T](it: TokenIterator,
    f: (TokenIterator) => T,
    sep: Token,
    end: Token)(implicit tct: ClassTag[T]): Array[T] = {
    val xs = new ArrayBuilder[T]()
    while (it.hasNext && it.head != end) {
      xs += f(it)
      if (it.head == sep)
        consumeToken(it)
    }
    xs.result()
  }

  def repUntil[T](it: TokenIterator,
    f: (TokenIterator) => T,
    end: Token)(implicit tct: ClassTag[T]): Array[T] = {
    val xs = new ArrayBuilder[T]()
    while (it.hasNext && it.head != end) {
      xs += f(it)
    }
    xs.result()
  }


  def base_seq_parser[T : ClassTag](f: TokenIterator => T)(it: TokenIterator): Array[T] = {
    punctuation(it, "(")
    val r = repUntil(it, f, PunctuationToken(")"))
    punctuation(it, ")")
    r
  }

  def decorator(it: TokenIterator): (String, String) = {
    punctuation(it, "@")
    val name = identifier(it)
    punctuation(it, "=")
    val desc = string_literal(it)
    (name, desc)
  }

  def ptuple_subset_field(env: TypeParserEnvironment)(it: TokenIterator): (Int, PType) = {
    val i = int32_literal(it)
    punctuation(it, ":")
    val t = ptype_expr(env)(it)
    i -> t
  }


  def tuple_subset_field(env: TypeParserEnvironment)(it: TokenIterator): (Int, Type) = {
    val i = int32_literal(it)
    punctuation(it, ":")
    val t = type_expr(env)(it)
    i -> t
  }

  def struct_field[T](f: TokenIterator => T)(it: TokenIterator): (String, T) = {
    val name = identifier(it)
    punctuation(it, ":")
    val typ = f(it)
    while (it.hasNext && it.head == PunctuationToken("@")) {
      decorator(it)
    }
    (name, typ)
  }

  def ptype_field(env: TypeParserEnvironment)(it: TokenIterator): (String, PType) = {
    struct_field(ptype_expr(env))(it)
  }

  def type_field(env: TypeParserEnvironment)(it: TokenIterator): (String, Type) = {
    struct_field(type_expr(env))(it)
  }

  def ptype_expr(env: TypeParserEnvironment)(it: TokenIterator): PType = {
    val req = it.head match {
      case x: PunctuationToken if x.value == "+" =>
        consumeToken(it)
        true
      case _ => false
    }

    val typ = identifier(it) match {
      case "PCInterval" =>
        punctuation(it, "[")
        val pointType = ptype_expr(env)(it)
        punctuation(it, "]")
        PCanonicalInterval(pointType, req)
      case "PBoolean" => PBoolean(req)
      case "PInt32" => PInt32(req)
      case "PInt64" => PInt64(req)
      case "PFloat32" => PFloat32(req)
      case "PFloat64" => PFloat64(req)
      case "PCBinary" => PCanonicalBinary(req)
      case "PCString" => PCanonicalString(req)
      case "PCLocus" =>
        punctuation(it, "(")
        val rg = identifier(it)
        punctuation(it, ")")
        PCanonicalLocus(env.getReferenceGenome(rg), req)
      case "PCCall" => PCanonicalCall(req)
      case "PCStream" =>
        punctuation(it, "[")
        val elementType = ptype_expr(env)(it)
        punctuation(it, "]")
        PCanonicalStream(elementType, req)
      case "PCArray" =>
        punctuation(it, "[")
        val elementType = ptype_expr(env)(it)
        punctuation(it, "]")
        PCanonicalArray(elementType, req)
      case "PCNDArray" =>
        punctuation(it, "[")
        val elementType = ptype_expr(env)(it)
        punctuation(it, ",")
        val nDims = int32_literal(it)
        punctuation(it, "]")
        PCanonicalNDArray(elementType, nDims, req)
      case "PCSet" =>
        punctuation(it, "[")
        val elementType = ptype_expr(env)(it)
        punctuation(it, "]")
        PCanonicalSet(elementType, req)
      case "PCDict" =>
        punctuation(it, "[")
        val keyType = ptype_expr(env)(it)
        punctuation(it, ",")
        val valueType = ptype_expr(env)(it)
        punctuation(it, "]")
        PCanonicalDict(keyType, valueType, req)
      case "PCTuple" =>
        punctuation(it, "[")
        val fields = repsepUntil(it, ptuple_subset_field(env), PunctuationToken(","), PunctuationToken("]"))
        punctuation(it, "]")
        PCanonicalTuple(fields.map { case (idx, t) => PTupleField(idx, t)}, req)
      case "PCStruct" =>
        punctuation(it, "{")
        val args = repsepUntil(it, ptype_field(env), PunctuationToken(","), PunctuationToken("}"))
        punctuation(it, "}")
        val fields = args.zipWithIndex.map { case ((id, t), i) => PField(id, t, i) }
        PCanonicalStruct(fields, req)
    }
    assert(typ.required == req)
    typ
  }

  def ptype_exprs(env: TypeParserEnvironment)(it: TokenIterator): Array[PType] =
    base_seq_parser(ptype_expr(env))(it)

  def type_exprs(env: TypeParserEnvironment)(it: TokenIterator): Array[Type] =
    base_seq_parser(type_expr(env))(it)

  def type_expr(env: TypeParserEnvironment)(it: TokenIterator): Type = {
    // skip requiredness token for back-compatibility
    it.head match {
      case x: PunctuationToken if x.value == "+" =>
        consumeToken(it)
      case _ =>
    }

    val typ = identifier(it) match {
      case "Interval" =>
        punctuation(it, "[")
        val pointType = type_expr(env)(it)
        punctuation(it, "]")
        TInterval(pointType)
      case "Boolean" => TBoolean
      case "Int32" => TInt32
      case "Int64" => TInt64
      case "Int" => TInt32
      case "Float32" => TFloat32
      case "Float64" => TFloat64
      case "String" => TString
      case "Locus" =>
        punctuation(it, "(")
        val rg = identifier(it)
        punctuation(it, ")")
        env.getReferenceGenome(rg).locusType
      case "Call" => TCall
      case "Stream" =>
        punctuation(it, "[")
        val elementType = type_expr(env)(it)
        punctuation(it, "]")
        TStream(elementType)
      case "Array" =>
        punctuation(it, "[")
        val elementType = type_expr(env)(it)
        punctuation(it, "]")
        TArray(elementType)
      case "NDArray" =>
        punctuation(it, "[")
        val elementType = type_expr(env)(it)
        punctuation(it, ",")
        val nDims = int32_literal(it)
        punctuation(it, "]")
        TNDArray(elementType, Nat(nDims))
      case "Set" =>
        punctuation(it, "[")
        val elementType = type_expr(env)(it)
        punctuation(it, "]")
        TSet(elementType)
      case "Dict" =>
        punctuation(it, "[")
        val keyType = type_expr(env)(it)
        punctuation(it, ",")
        val valueType = type_expr(env)(it)
        punctuation(it, "]")
        TDict(keyType, valueType)
      case "Tuple" =>
        punctuation(it, "[")
        val types = repsepUntil(it, type_expr(env), PunctuationToken(","), PunctuationToken("]"))
        punctuation(it, "]")
        TTuple(types: _*)
      case "TupleSubset" =>
        punctuation(it, "[")
        val fields = repsepUntil(it, tuple_subset_field(env), PunctuationToken(","), PunctuationToken("]"))
        punctuation(it, "]")
        TTuple(fields.map { case (idx, t) => TupleField(idx, t)})
      case "Struct" =>
        punctuation(it, "{")
        val args = repsepUntil(it, type_field(env), PunctuationToken(","), PunctuationToken("}"))
        punctuation(it, "}")
        val fields = args.zipWithIndex.map { case ((id, t), i) => Field(id, t, i) }
        TStruct(fields)
      case "Union" =>
        punctuation(it, "{")
        val args = repsepUntil(it, type_field(env), PunctuationToken(","), PunctuationToken("}"))
        punctuation(it, "}")
        val cases = args.zipWithIndex.map { case ((id, t), i) => Case(id, t, i) }
        TUnion(cases)
      case "Void" => TVoid
      case "Shuffle" =>
        punctuation(it, "{")
        val keyFields = sort_fields(it)
        punctuation(it, ",")
        val rowType = type_expr(env)(it).asInstanceOf[TStruct]
        punctuation(it, ",")
        val rowEType = EType.eTypeParser(it).asInstanceOf[EBaseStruct]
        punctuation(it, ",")
        val keyEType = EType.eTypeParser(it).asInstanceOf[EBaseStruct]
        punctuation(it, "}")
        TShuffle(keyFields, rowType, rowEType, keyEType)
    }
    typ
  }

  def sort_fields(it: TokenIterator): Array[SortField] =
    base_seq_parser(sort_field)(it)

  def sort_field(it: TokenIterator): SortField = {
    val sortField = identifier(it)
    val field = sortField.substring(1)
    val sortOrder = SortOrder.parse(sortField.substring(0, 1))
    SortField(field, sortOrder)
  }

  def keys(it: TokenIterator): Array[String] = {
    punctuation(it, "[")
    val keys = repsepUntil(it, identifier, PunctuationToken(","), PunctuationToken("]"))
    punctuation(it, "]")
    keys
  }

  def trailing_keys(it: TokenIterator): Array[String] = {
    it.head match {
      case x: PunctuationToken if x.value == "]" =>
        Array.empty[String]
      case x: PunctuationToken if x.value == "," =>
        punctuation(it, ",")
        repsepUntil(it, identifier, PunctuationToken(","), PunctuationToken("]"))
    }
  }

  def rvd_type_expr(env: TypeParserEnvironment)(it: TokenIterator): RVDType = {
    identifier(it) match {
      case "RVDType" | "OrderedRVDType" =>
        punctuation(it, "{")
        identifier(it, "key")
        punctuation(it, ":")
        punctuation(it, "[")
        val partitionKey = keys(it)
        val restKey = trailing_keys(it)
        punctuation(it, "]")
        punctuation(it, ",")
        identifier(it, "row")
        punctuation(it, ":")
        val rowType = coerce[PStruct](ptype_expr(env)(it))
        RVDType(rowType, partitionKey ++ restKey)
    }
  }

  def table_type_expr(env: TypeParserEnvironment)(it: TokenIterator): TableType = {
    identifier(it, "Table")
    punctuation(it, "{")

    identifier(it, "global")
    punctuation(it, ":")
    val globalType = coerce[TStruct](type_expr(env)(it))
    punctuation(it, ",")

    identifier(it, "key")
    punctuation(it, ":")
    val key = opt(it, keys).getOrElse(Array.empty[String])
    punctuation(it, ",")

    identifier(it, "row")
    punctuation(it, ":")
    val rowType = coerce[TStruct](type_expr(env)(it))
    punctuation(it, "}")
    TableType(rowType, key.toFastIndexedSeq, coerce[TStruct](globalType))
  }

  def matrix_type_expr(env: TypeParserEnvironment)(it: TokenIterator): MatrixType = {
    identifier(it, "Matrix")
    punctuation(it, "{")

    identifier(it, "global")
    punctuation(it, ":")
    val globalType = coerce[TStruct](type_expr(env)(it))
    punctuation(it, ",")

    identifier(it, "col_key")
    punctuation(it, ":")
    val colKey = keys(it)
    punctuation(it, ",")

    identifier(it, "col")
    punctuation(it, ":")
    val colType = coerce[TStruct](type_expr(env)(it))
    punctuation(it, ",")

    identifier(it, "row_key")
    punctuation(it, ":")
    punctuation(it, "[")
    val rowPartitionKey = keys(it)
    val rowRestKey = trailing_keys(it)
    punctuation(it, "]")
    punctuation(it, ",")

    identifier(it, "row")
    punctuation(it, ":")
    val rowType = coerce[TStruct](type_expr(env)(it))
    punctuation(it, ",")

    identifier(it, "entry")
    punctuation(it, ":")
    val entryType = coerce[TStruct](type_expr(env)(it))
    punctuation(it, "}")

    MatrixType(coerce[TStruct](globalType), colKey, colType, rowPartitionKey ++ rowRestKey, rowType, entryType)
  }

  def agg_op(it: TokenIterator): AggOp =
    AggOp.fromString(identifier(it))

  def agg_state_signature(env: TypeParserEnvironment)(it: TokenIterator): AggStateSig = {
    punctuation(it, "(")
    val sig = identifier(it) match {
      case "TypedStateSig" =>
        val pt = ptype_expr(env)(it)
        TypedStateSig(pt)
      case "DownsampleStateSig" =>
        val labelType = ptype_expr(env)(it)
        DownsampleStateSig(coerce[PArray](labelType))
      case "TakeStateSig" =>
        val pt = ptype_expr(env)(it)
        TakeStateSig(pt)
      case "TakeByStateSig" =>
        val vt = ptype_expr(env)(it)
        val kt = ptype_expr(env)(it)
        TakeByStateSig(vt, kt, Ascending)
      case "CollectStateSig" =>
        val pt = ptype_expr(env)(it)
        CollectStateSig(pt)
      case "CollectAsSetStateSig" =>
        val pt = ptype_expr(env)(it)
        CollectAsSetStateSig(pt)
      case "CallStatsStateSig" => CallStatsStateSig()
      case "ArrayAggStateSig" =>
        val nested = agg_state_signatures(env)(it)
        ArrayAggStateSig(nested)
      case "GroupedStateSig" =>
        val kt = ptype_expr(env)(it)
        val nested = agg_state_signatures(env)(it)
        GroupedStateSig(kt, nested)
      case "ApproxCDFStateSig" => ApproxCDFStateSig()
    }
    punctuation(it, ")")
    sig
  }

  def agg_state_signatures(env: TypeParserEnvironment)(it: TokenIterator): Array[AggStateSig] =
    base_seq_parser(agg_state_signature(env))(it)

  def p_agg_sigs(env: TypeParserEnvironment)(it: TokenIterator): Array[PhysicalAggSig] =
    base_seq_parser(p_agg_sig(env))(it)

  def p_agg_sig(env: TypeParserEnvironment)(it: TokenIterator): PhysicalAggSig = {
    punctuation(it, "(")
    val sig = identifier(it) match {
      case "Grouped" =>
        val pt = ptype_expr(env)(it)
        val nested = p_agg_sigs(env)(it)
        GroupedAggSig(pt, nested)
      case "ArrayLen" =>
        val knownLength = boolean_literal(it)
        val nested = p_agg_sigs(env)(it)
        ArrayLenAggSig(knownLength, nested)
      case "AggElements" =>
        val nested = p_agg_sigs(env)(it)
        AggElementsAggSig(nested)
      case op =>
        val state = agg_state_signature(env)(it)
        PhysicalAggSig(AggOp.fromString(op), state)
    }
    punctuation(it, ")")
    sig
  }

  def agg_signature(env: TypeParserEnvironment)(it: TokenIterator): AggSignature = {
    punctuation(it, "(")
    val op = agg_op(it)
    val initArgs = type_exprs(env)(it)
    val seqOpArgs = type_exprs(env)(it)
    punctuation(it, ")")
    AggSignature(op, initArgs, seqOpArgs)
  }

  def agg_signatures(env: TypeParserEnvironment)(it: TokenIterator): Array[AggSignature] =
    base_seq_parser(agg_signature(env))(it)

  def ir_value(env: TypeParserEnvironment)(it: TokenIterator): (Type, Any) = {
    val typ = type_expr(env)(it)
    val s = string_literal(it)
    val vJSON = JsonMethods.parse(s)
    val v = JSONAnnotationImpex.importAnnotation(vJSON, typ)
    (typ, v)
  }

  def named_value_irs(env: IRParserEnvironment)(it: TokenIterator): Array[(String, IR)] =
    repUntil(it, named_value_ir(env), PunctuationToken(")"))

  def named_value_ir(env: IRParserEnvironment)(it: TokenIterator): (String, IR) = {
    punctuation(it, "(")
    val name = identifier(it)
    val value = ir_value_expr(env)(it)
    punctuation(it, ")")
    (name, value)
  }

  def ir_value_exprs(env: IRParserEnvironment)(it: TokenIterator): Array[IR] = {
    punctuation(it, "(")
    val irs = ir_value_children(env)(it)
    punctuation(it, ")")
    irs
  }

  def ir_value_children(env: IRParserEnvironment)(it: TokenIterator): Array[IR] =
    repUntil(it, ir_value_expr(env), PunctuationToken(")"))

  def ir_value_expr(env: IRParserEnvironment)(it: TokenIterator): IR = {
    punctuation(it, "(")
    val ir = ir_value_expr_1(env)(it)
    punctuation(it, ")")
    ir
  }

  def ir_value_expr_1(env: IRParserEnvironment)(it: TokenIterator): IR = {
    identifier(it) match {
      case "I32" => I32(int32_literal(it))
      case "I64" => I64(int64_literal(it))
      case "F32" => F32(float32_literal(it))
      case "F64" => F64(float64_literal(it))
      case "Str" => Str(string_literal(it))
      case "UUID4" => UUID4(identifier(it))
      case "True" => True()
      case "False" => False()
      case "Literal" =>
        val (t, v) = ir_value(env.typEnv)(it)
        Literal.coerce(t, v)
      case "Void" => Void()
      case "Cast" =>
        val typ = type_expr(env.typEnv)(it)
        val v = ir_value_expr(env)(it)
        Cast(v, typ)
      case "CastRename" =>
        val typ = type_expr(env.typEnv)(it)
        val v = ir_value_expr(env)(it)
        CastRename(v, typ)
      case "NA" => NA(type_expr(env.typEnv)(it))
      case "IsNA" => IsNA(ir_value_expr(env)(it))
      case "Coalesce" =>
        val children = ir_value_children(env)(it)
        require(children.nonEmpty)
        Coalesce(children)
      case "If" =>
        val cond = ir_value_expr(env)(it)
        val consq = ir_value_expr(env)(it)
        val altr = ir_value_expr(env)(it)
        If(cond, consq, altr)
      case "Let" =>
        val name = identifier(it)
        val value = ir_value_expr(env)(it)
        val body = ir_value_expr(env + (name -> value.typ))(it)
        Let(name, value, body)
      case "AggLet" =>
        val name = identifier(it)
        val isScan = boolean_literal(it)
        val value = ir_value_expr(env)(it)
        val body = ir_value_expr(env + (name -> value.typ))(it)
        AggLet(name, value, body, isScan)
      case "TailLoop" =>
        val name = identifier(it)
        val paramNames = identifiers(it)
        val params = paramNames.map { n => n -> ir_value_expr(env)(it) }
        val bodyEnv = env.update(params.map { case (n, v) => n -> v.typ}.toMap)
        val body = ir_value_expr(bodyEnv)(it)
        TailLoop(name, params, body)
      case "Recur" =>
        val name = identifier(it)
        val typ = type_expr(env.typEnv)(it)
        val args = ir_value_children(env)(it)
        Recur(name, args, typ)
      case "Ref" =>
        val id = identifier(it)
        Ref(id, env.refMap(id))
      case "RelationalRef" =>
        val id = identifier(it)
        val t = type_expr(env.typEnv)(it)
        RelationalRef(id, t)
      case "RelationalLet" =>
        val name = identifier(it)
        val value = ir_value_expr(env)(it)
        val body = ir_value_expr(env + (name -> value.typ))(it)
        RelationalLet(name, value, body)
      case "ApplyBinaryPrimOp" =>
        val op = BinaryOp.fromString(identifier(it))
        val l = ir_value_expr(env)(it)
        val r = ir_value_expr(env)(it)
        ApplyBinaryPrimOp(op, l, r)
      case "ApplyUnaryPrimOp" =>
        val op = UnaryOp.fromString(identifier(it))
        val x = ir_value_expr(env)(it)
        ApplyUnaryPrimOp(op, x)
      case "ApplyComparisonOp" =>
        val opName = identifier(it)
        val l = ir_value_expr(env)(it)
        val r = ir_value_expr(env)(it)
        val op = ComparisonOp.fromStringAndTypes((opName, l.typ, r.typ))
        ApplyComparisonOp(op, l, r)
      case "MakeArray" =>
        val typ = opt(it, type_expr(env.typEnv)).map(_.asInstanceOf[TArray]).orNull
        val args = ir_value_children(env)(it)
        MakeArray.unify(args, typ)
      case "MakeStream" =>
        val typ = opt(it, type_expr(env.typEnv)).map(_.asInstanceOf[TStream]).orNull
        val args = ir_value_children(env)(it)
        MakeStream(args, typ)
      case "ArrayRef" =>
        val a = ir_value_expr(env)(it)
        val i = ir_value_expr(env)(it)
        val s = ir_value_expr(env)(it)
        ArrayRef(a, i, s)
      case "ArrayLen" => ArrayLen(ir_value_expr(env)(it))
      case "StreamLen" => StreamLen(ir_value_expr(env)(it))
      case "StreamRange" =>
        val start = ir_value_expr(env)(it)
        val stop = ir_value_expr(env)(it)
        val step = ir_value_expr(env)(it)
        StreamRange(start, stop, step)
      case "ArrayZeros" => ArrayZeros(ir_value_expr(env)(it))
      case "ArraySort" =>
        val l = identifier(it)
        val r = identifier(it)
        val a = ir_value_expr(env)(it)
        val elt = coerce[TStream](a.typ).elementType
        val lessThan = ir_value_expr(env + (l -> elt) + (r -> elt))(it)
        ArraySort(a, l, r, lessThan)
      case "MakeNDArray" =>
        val data = ir_value_expr(env)(it)
        val shape = ir_value_expr(env)(it)
        val rowMajor = ir_value_expr(env)(it)
        MakeNDArray(data, shape, rowMajor)
      case "NDArrayShape" =>
        val nd = ir_value_expr(env)(it)
        NDArrayShape(nd)
      case "NDArrayReshape" =>
        val nd = ir_value_expr(env)(it)
        val shape = ir_value_expr(env)(it)
        NDArrayReshape(nd, shape)
      case "NDArrayConcat" =>
        val axis = int32_literal(it)
        val nds = ir_value_expr(env)(it)
        NDArrayConcat(nds, axis)
      case "NDArrayMap" =>
        val name = identifier(it)
        val nd = ir_value_expr(env)(it)
        val body = ir_value_expr(env + (name -> coerce[TNDArray](nd.typ).elementType))(it)
        NDArrayMap(nd, name, body)
      case "NDArrayMap2" =>
        val lName = identifier(it)
        val rName = identifier(it)
        val l = ir_value_expr(env)(it)
        val r = ir_value_expr(env)(it)
        val body_env = (env + (lName -> coerce[TNDArray](l.typ).elementType)
                            + (rName -> coerce[TNDArray](r.typ).elementType))
        val body = ir_value_expr(body_env)(it)
        NDArrayMap2(l, r, lName, rName, body)
      case "NDArrayReindex" =>
        val indexExpr = int32_literals(it)
        val nd = ir_value_expr(env)(it)
        NDArrayReindex(nd, indexExpr)
      case "NDArrayAgg" =>
        val axes = int32_literals(it)
        val nd = ir_value_expr(env)(it)
        NDArrayAgg(nd, axes)
      case "NDArrayRef" =>
        val nd = ir_value_expr(env)(it)
        val idxs = ir_value_children(env)(it)
        NDArrayRef(nd, idxs)
      case "NDArraySlice" =>
        val nd = ir_value_expr(env)(it)
        val slices = ir_value_expr(env)(it)
        NDArraySlice(nd, slices)
      case "NDArrayFilter" =>
        val nd = ir_value_expr(env)(it)
        val filters = Array.fill(coerce[TNDArray](nd.typ).nDims)(ir_value_expr(env)(it))
        NDArrayFilter(nd, filters.toFastIndexedSeq)
      case "NDArrayMatMul" =>
        val l = ir_value_expr(env)(it)
        val r = ir_value_expr(env)(it)
        NDArrayMatMul(l, r)
      case "NDArrayWrite" =>
        val nd = ir_value_expr(env)(it)
        val path = ir_value_expr(env)(it)
        NDArrayWrite(nd, path)
      case "NDArrayQR" =>
        val mode = string_literal(it)
        val nd = ir_value_expr(env)(it)
        NDArrayQR(nd, mode)
      case "NDArrayInv" =>
        val nd = ir_value_expr(env)(it)
        NDArrayInv(nd)
      case "ToSet" => ToSet(ir_value_expr(env)(it))
      case "ToDict" => ToDict(ir_value_expr(env)(it))
      case "ToArray" => ToArray(ir_value_expr(env)(it))
      case "CastToArray" => CastToArray(ir_value_expr(env)(it))
      case "ToStream" => ToStream(ir_value_expr(env)(it))
      case "LowerBoundOnOrderedCollection" =>
        val onKey = boolean_literal(it)
        val col = ir_value_expr(env)(it)
        val elem = ir_value_expr(env)(it)
        LowerBoundOnOrderedCollection(col, elem, onKey)
      case "GroupByKey" =>
        val col = ir_value_expr(env)(it)
        GroupByKey(col)
      case "StreamMap" =>
        val name = identifier(it)
        val a = ir_value_expr(env)(it)
        val body = ir_value_expr(env + (name -> coerce[TStream](a.typ).elementType))(it)
        StreamMap(a, name, body)
      case "StreamTake" =>
        val a = ir_value_expr(env)(it)
        val num = ir_value_expr(env)(it)
        StreamTake(a, num)
      case "StreamDrop" =>
        val a = ir_value_expr(env)(it)
        val num = ir_value_expr(env)(it)
        StreamDrop(a, num)
      case "StreamMerge" =>
        val key = identifiers(it)
        val left = ir_value_expr(env)(it)
        val right = ir_value_expr(env)(it)
        StreamMerge(left, right, key)
      case "StreamZip" =>
        val behavior = identifier(it) match {
          case "AssertSameLength" => ArrayZipBehavior.AssertSameLength
          case "TakeMinLength" => ArrayZipBehavior.TakeMinLength
          case "ExtendNA" => ArrayZipBehavior.ExtendNA
          case "AssumeSameLength" => ArrayZipBehavior.AssumeSameLength
        }
        val names = identifiers(it)
        val as = names.map(_ => ir_value_expr(env)(it))
        val body = ir_value_expr(env ++ names.zip(as.map(a => coerce[TStream](a.typ).elementType)))(it)
        StreamZip(as, names, body, behavior)
      case "StreamFilter" =>
        val name = identifier(it)
        val a = ir_value_expr(env)(it)
        val body = ir_value_expr(env + (name -> coerce[TStream](a.typ).elementType))(it)
        StreamFilter(a, name, body)
      case "StreamFlatMap" =>
        val name = identifier(it)
        val a = ir_value_expr(env)(it)
        val body = ir_value_expr(env + (name -> coerce[TStream](a.typ).elementType))(it)
        StreamFlatMap(a, name, body)
      case "StreamFold" =>
        val accumName = identifier(it)
        val valueName = identifier(it)
        val a = ir_value_expr(env)(it)
        val zero = ir_value_expr(env)(it)
        val eltType = coerce[TStream](a.typ).elementType
        val body = ir_value_expr(env.update(Map(accumName -> zero.typ, valueName -> eltType)))(it)
        StreamFold(a, zero, accumName, valueName, body)
      case "StreamFold2" =>
        val accumNames = identifiers(it)
        val valueName = identifier(it)
        val a = ir_value_expr(env)(it)
        val accs = accumNames.map(name => (name, ir_value_expr(env)(it)))
        val eltType = coerce[TStream](a.typ).elementType
        val resultEnv = env.update(accs.map { case (name, value) => (name, value.typ) }.toMap)
        val seqEnv = resultEnv.update(Map(valueName -> eltType))
        val seqs = Array.tabulate(accs.length)(_ => ir_value_expr(seqEnv)(it))
        val res = ir_value_expr(resultEnv)(it)
        StreamFold2(a, accs, valueName, seqs, res)
      case "StreamScan" =>
        val accumName = identifier(it)
        val valueName = identifier(it)
        val a = ir_value_expr(env)(it)
        val zero = ir_value_expr(env)(it)
        val eltType = coerce[TStream](a.typ).elementType
        val body = ir_value_expr(env.update(Map(accumName -> zero.typ, valueName -> eltType)))(it)
        StreamScan(a, zero, accumName, valueName, body)
      case "StreamJoinRightDistinct" =>
        val lKey = identifiers(it)
        val rKey = identifiers(it)
        val l = identifier(it)
        val r = identifier(it)
        val joinType = identifier(it)
        val left = ir_value_expr(env)(it)
        val right = ir_value_expr(env)(it)
        val lelt = coerce[TStream](left.typ).elementType
        val relt = coerce[TStream](right.typ).elementType
        val join = ir_value_expr(env.update(Map(l -> lelt, r -> relt)))(it)
        StreamJoinRightDistinct(left, right, lKey, rKey, l, r, join, joinType)
      case "StreamFor" =>
        val name = identifier(it)
        val a = ir_value_expr(env)(it)
        val body = ir_value_expr(env + (name -> coerce[TStream](a.typ).elementType))(it)
        StreamFor(a, name, body)
      case "StreamAgg" =>
        val name = identifier(it)
        val a = ir_value_expr(env)(it)
        val query = ir_value_expr(env + (name -> coerce[TStream](a.typ).elementType))(it)
        StreamAgg(a, name, query)
      case "StreamAggScan" =>
        val name = identifier(it)
        val a = ir_value_expr(env)(it)
        val query = ir_value_expr(env + (name -> coerce[TStream](a.typ).elementType))(it)
        StreamAggScan(a, name, query)
      case "RunAgg" =>
        val signatures = agg_state_signatures(env.typEnv)(it)
        val body = ir_value_expr(env)(it)
        val result = ir_value_expr(env)(it)
        RunAgg(body, result, signatures)
      case "RunAggScan" =>
        val name = identifier(it)
        val signatures = agg_state_signatures(env.typEnv)(it)
        val array = ir_value_expr(env)(it)
        val newE = env + (name -> coerce[TStream](array.typ).elementType)
        val init = ir_value_expr(env)(it)
        val seq = ir_value_expr(newE)(it)
        val result = ir_value_expr(newE)(it)
        RunAggScan(array, name, init, seq, result, signatures)
      case "AggFilter" =>
        val isScan = boolean_literal(it)
        val cond = ir_value_expr(env)(it)
        val aggIR = ir_value_expr(env)(it)
        AggFilter(cond, aggIR, isScan)
      case "AggExplode" =>
        val name = identifier(it)
        val isScan = boolean_literal(it)
        val a = ir_value_expr(env)(it)
        val aggBody = ir_value_expr(env + (name -> coerce[TStream](a.typ).elementType))(it)
        AggExplode(a, name, aggBody, isScan)
      case "AggGroupBy" =>
        val isScan = boolean_literal(it)
        val key = ir_value_expr(env)(it)
        val aggIR = ir_value_expr(env)(it)
        AggGroupBy(key, aggIR, isScan)
      case "AggArrayPerElement" =>
        val elementName = identifier(it)
        val indexName = identifier(it)
        val isScan = boolean_literal(it)
        val hasKnownLength = boolean_literal(it)
        val a = ir_value_expr(env)(it)
        val aggBody = ir_value_expr(env
          + (elementName -> coerce[TArray](a.typ).elementType)
          + (indexName -> TInt32))(it)
        val knownLength = if (hasKnownLength) Some(ir_value_expr(env)(it)) else None
        AggArrayPerElement(a, elementName, indexName, aggBody, knownLength, isScan)
      case "ApplyAggOp" =>
        val aggOp = agg_op(it)
        val initOpArgs = ir_value_exprs(env)(it)
        val seqOpArgs = ir_value_exprs(env)(it)
        val aggSig = AggSignature(aggOp, initOpArgs.map(arg => arg.typ), seqOpArgs.map(arg => arg.typ))
        ApplyAggOp(initOpArgs, seqOpArgs, aggSig)
      case "ApplyScanOp" =>
        val aggOp = agg_op(it)
        val initOpArgs = ir_value_exprs(env)(it)
        val seqOpArgs = ir_value_exprs(env)(it)
        val aggSig = AggSignature(aggOp, initOpArgs.map(arg => arg.typ), seqOpArgs.map(arg => arg.typ))
        ApplyScanOp(initOpArgs, seqOpArgs, aggSig)
      case "InitOp" =>
        val i = int32_literal(it)
        val aggSig = p_agg_sig(env.typEnv)(it)
        val args = ir_value_exprs(env)(it)
        InitOp(i, args, aggSig)
      case "SeqOp" =>
        val i = int32_literal(it)
        val aggSig = p_agg_sig(env.typEnv)(it)
        val args = ir_value_exprs(env)(it)
        SeqOp(i, args, aggSig)
      case "CombOp" =>
        val i1 = int32_literal(it)
        val i2 = int32_literal(it)
        val aggSig = p_agg_sig(env.typEnv)(it)
        CombOp(i1, i2, aggSig)
      case "ResultOp" =>
        val i = int32_literal(it)
        val aggSigs = p_agg_sigs(env.typEnv)(it)
        ResultOp(i, aggSigs)
      case "AggStateValue" =>
        val i = int32_literal(it)
        val sig = agg_state_signature(env.typEnv)(it)
        AggStateValue(i, sig)
      case "InitFromSerializedValue" =>
        val i = int32_literal(it)
        val sig = agg_state_signature(env.typEnv)(it)
        val value = ir_value_expr(env)(it)
        InitFromSerializedValue(i, value, sig)
      case "CombOpValue" =>
        val i = int32_literal(it)
        val sig = p_agg_sig(env.typEnv)(it)
        val value = ir_value_expr(env)(it)
        CombOpValue(i, value, sig)
      case "SerializeAggs" =>
        val i = int32_literal(it)
        val i2 = int32_literal(it)
        val spec = BufferSpec.parse(string_literal(it))
        val aggSigs = agg_state_signatures(env.typEnv)(it)
        SerializeAggs(i, i2, spec, aggSigs)
      case "DeserializeAggs" =>
        val i = int32_literal(it)
        val i2 = int32_literal(it)
        val spec = BufferSpec.parse(string_literal(it))
        val aggSigs = agg_state_signatures(env.typEnv)(it)
        DeserializeAggs(i, i2, spec, aggSigs)
      case "Begin" =>
        val xs = ir_value_children(env)(it)
        Begin(xs)
      case "MakeStruct" =>
        val fields = named_value_irs(env)(it)
        MakeStruct(fields)
      case "SelectFields" =>
        val fields = identifiers(it)
        val old = ir_value_expr(env)(it)
        SelectFields(old, fields)
      case "InsertFields" =>
        val old = ir_value_expr(env)(it)
        val fieldOrder = opt(it, string_literals)
        val fields = named_value_irs(env)(it)
        InsertFields(old, fields, fieldOrder.map(_.toFastIndexedSeq))
      case "GetField" =>
        val name = identifier(it)
        val s = ir_value_expr(env)(it)
        GetField(s, name)
      case "MakeTuple" =>
        val indices = int32_literals(it)
        val args = ir_value_children(env)(it)
        MakeTuple(indices.zip(args))
      case "GetTupleElement" =>
        val idx = int32_literal(it)
        val tuple = ir_value_expr(env)(it)
        GetTupleElement(tuple, idx)
      case "In" =>
        val typ = ptype_expr(env.typEnv)(it)
        val idx = int32_literal(it)
        In(idx, typ)
      case "Die" =>
        val typ = type_expr(env.typEnv)(it)
        val msg = ir_value_expr(env)(it)
        Die(msg, typ)
      case "ApplySeeded" =>
        val function = identifier(it)
        val seed = int64_literal(it)
        val rt = type_expr(env.typEnv)(it)
        val args = ir_value_children(env)(it)
        ApplySeeded(function, args, seed, rt)
      case "ApplyIR" | "ApplySpecial" | "Apply" =>
        val function = identifier(it)
        val typeArgs = type_exprs(env.typEnv)(it)
        val rt = type_expr(env.typEnv)(it)
        val args = ir_value_children(env)(it)
        invoke(function, rt, typeArgs, args: _*)
      case "MatrixCount" =>
        val child = matrix_ir(env.withRefMap(Map.empty))(it)
        MatrixCount(child)
      case "TableCount" =>
        val child = table_ir(env.withRefMap(Map.empty))(it)
        TableCount(child)
      case "TableGetGlobals" =>
        val child = table_ir(env.withRefMap(Map.empty))(it)
        TableGetGlobals(child)
      case "TableCollect" =>
        val child = table_ir(env.withRefMap(Map.empty))(it)
        TableCollect(child)
      case "TableAggregate" =>
        val child = table_ir(env.withRefMap(Map.empty))(it)
        val query = ir_value_expr(env.update(child.typ.refMap))(it)
        TableAggregate(child, query)
      case "TableToValueApply" =>
        val config = string_literal(it)
        val child = table_ir(env)(it)
        TableToValueApply(child, RelationalFunctions.lookupTableToValue(env.ctx, config))
      case "MatrixToValueApply" =>
        val config = string_literal(it)
        val child = matrix_ir(env)(it)
        MatrixToValueApply(child, RelationalFunctions.lookupMatrixToValue(env.ctx, config))
      case "BlockMatrixToValueApply" =>
        val config = string_literal(it)
        val child = blockmatrix_ir(env)(it)
        BlockMatrixToValueApply(child, RelationalFunctions.lookupBlockMatrixToValue(env.ctx, config))
      case "BlockMatrixCollect" =>
        val child = blockmatrix_ir(env)(it)
        BlockMatrixCollect(child)
      case "TableWrite" =>
        implicit val formats = TableWriter.formats
        val writerStr = string_literal(it)
        val child = table_ir(env)(it)
        TableWrite(child, deserialize[TableWriter](writerStr))
      case "TableMultiWrite" =>
        implicit val formats = WrappedMatrixNativeMultiWriter.formats
        val writerStr = string_literal(it)
        val children = table_ir_children(env)(it)
        TableMultiWrite(children, deserialize[WrappedMatrixNativeMultiWriter](writerStr))
      case "MatrixAggregate" =>
        val child = matrix_ir(env.withRefMap(Map.empty))(it)
        val query = ir_value_expr(env.update(child.typ.refMap))(it)
        MatrixAggregate(child, query)
      case "MatrixWrite" =>
        val writerStr = string_literal(it)
        implicit val formats: Formats = MatrixWriter.formats
        val writer = deserialize[MatrixWriter](writerStr)
        val child = matrix_ir(env.withRefMap(Map.empty))(it)
        MatrixWrite(child, writer)
      case "MatrixMultiWrite" =>
        val writerStr = string_literal(it)
        implicit val formats = MatrixNativeMultiWriter.formats
        val writer = deserialize[MatrixNativeMultiWriter](writerStr)
        val children = matrix_ir_children(env)(it)
        MatrixMultiWrite(children, writer)
      case "BlockMatrixWrite" =>
        val writerStr = string_literal(it)
        implicit val formats: Formats = BlockMatrixWriter.formats
        val writer = deserialize[BlockMatrixWriter](writerStr)
        val child = blockmatrix_ir(env)(it)
        BlockMatrixWrite(child, writer)
      case "BlockMatrixMultiWrite" =>
        val writerStr = string_literal(it)
        implicit val formats: Formats = BlockMatrixWriter.formats
        val writer = deserialize[BlockMatrixMultiWriter](writerStr)
        val blockMatrices = repUntil(it, blockmatrix_ir(env), PunctuationToken(")"))
        BlockMatrixMultiWrite(blockMatrices.toFastIndexedSeq, writer)
      case "UnpersistBlockMatrix" =>
        UnpersistBlockMatrix(blockmatrix_ir(env)(it))
      case "CollectDistributedArray" =>
        val cname = identifier(it)
        val gname = identifier(it)
        val ctxs = ir_value_expr(env)(it)
        val globals = ir_value_expr(env)(it)
        val body = ir_value_expr(env + (cname -> coerce[TStream](ctxs.typ).elementType) + (gname -> globals.typ))(it)
        CollectDistributedArray(ctxs, globals, cname, gname, body)
      case "JavaIR" =>
        val name = identifier(it)
        env.irMap(name).asInstanceOf[IR]
      case "ReadPartition" =>
        val rowType = coerce[TStruct](type_expr(env.typEnv)(it))
        import PartitionReader.formats
        val reader = JsonMethods.parse(string_literal(it)).extract[PartitionReader]
        val context = ir_value_expr(env)(it)
        ReadPartition(context, rowType, reader)
      case "WritePartition" =>
        import PartitionWriter.formats
        val writer = JsonMethods.parse(string_literal(it)).extract[PartitionWriter]
        val stream = ir_value_expr(env)(it)
        val ctx = ir_value_expr(env)(it)
        WritePartition(stream, ctx, writer)
      case "WriteMetadata" =>
        import MetadataWriter.formats
        val writer = JsonMethods.parse(string_literal(it)).extract[MetadataWriter]
        val ctx = ir_value_expr(env)(it)
        WriteMetadata(ctx, writer)
      case "ReadValue" =>
        import AbstractRVDSpec.formats
        val spec = JsonMethods.parse(string_literal(it)).extract[AbstractTypedCodecSpec]
        val typ = type_expr(env.typEnv)(it)
        val path = ir_value_expr(env)(it)
        ReadValue(path, spec, typ)
      case "WriteValue" =>
        import AbstractRVDSpec.formats
        val spec = JsonMethods.parse(string_literal(it)).extract[AbstractTypedCodecSpec]
        val value = ir_value_expr(env)(it)
        val path = ir_value_expr(env)(it)
        WriteValue(value, path, spec)
      case "LiftMeOut" =>
        LiftMeOut(ir_value_expr(env)(it))
      case "ReadPartition" =>
        val rowType = coerce[TStruct](type_expr(env.typEnv)(it))
        import PartitionReader.formats
        val reader = JsonMethods.parse(string_literal(it)).extract[PartitionReader]
        val context = ir_value_expr(env)(it)
        ReadPartition(context, rowType, reader)
      case "ShuffleWith" =>
        val shuffleType = coerce[TShuffle](type_expr(env.typEnv)(it))
        val name = identifier(it)
        val writer = ir_value_expr(env + (name -> shuffleType))(it)
        val readers = ir_value_expr(env + (name -> shuffleType))(it)
        ShuffleWith(
          shuffleType.keyFields, shuffleType.rowType, shuffleType.rowEType, shuffleType.keyEType,
          name, writer, readers)
      case "ShuffleWrite" =>
        val id = ir_value_expr(env)(it)
        val rows = ir_value_expr(env)(it)
        ShuffleWrite(id, rows)
      case "ShufflePartitionBounds" =>
        val id = ir_value_expr(env)(it)
        val nPartitions = ir_value_expr(env)(it)
        ShufflePartitionBounds(id, nPartitions)
      case "ShuffleRead" =>
        val id = ir_value_expr(env)(it)
        val keyRange = ir_value_expr(env)(it)
        ShuffleRead(id, keyRange)
    }
  }

  def table_irs(env: IRParserEnvironment)(it: TokenIterator): Array[TableIR] = {
    punctuation(it, "(")
    val tirs = table_ir_children(env)(it)
    punctuation(it, ")")
    tirs
  }

  def table_ir_children(env: IRParserEnvironment)(it: TokenIterator): Array[TableIR] =
    repUntil(it, table_ir(env), PunctuationToken(")"))

  def table_ir(env: IRParserEnvironment)(it: TokenIterator): TableIR = {
    punctuation(it, "(")
    val ir = table_ir_1(env)(it)
    punctuation(it, ")")
    ir
  }

  def table_ir_1(env: IRParserEnvironment)(it: TokenIterator): TableIR = {
    // FIXME TableImport
    identifier(it) match {
      case "TableKeyBy" =>
        val keys = identifiers(it)
        val isSorted = boolean_literal(it)
        val child = table_ir(env)(it)
        TableKeyBy(child, keys, isSorted)
      case "TableDistinct" =>
        val child = table_ir(env)(it)
        TableDistinct(child)
      case "TableFilter" =>
        val child = table_ir(env)(it)
        val pred = ir_value_expr(env.withRefMap(child.typ.refMap))(it)
        TableFilter(child, pred)
      case "TableRead" =>
        val requestedType = opt(it, table_type_expr(env.typEnv))
        val dropRows = boolean_literal(it)
        val readerStr = string_literal(it)
        val reader = TableReader.fromJValue(env.ctx.fs, JsonMethods.parse(readerStr).asInstanceOf[JObject])
        TableRead(requestedType.getOrElse(reader.fullType), dropRows, reader)
      case "MatrixColsTable" =>
        val child = matrix_ir(env)(it)
        MatrixColsTable(child)
      case "MatrixRowsTable" =>
        val child = matrix_ir(env)(it)
        MatrixRowsTable(child)
      case "MatrixEntriesTable" =>
        val child = matrix_ir(env)(it)
        MatrixEntriesTable(child)
      case "TableAggregateByKey" =>
        val child = table_ir(env)(it)
        val expr = ir_value_expr(env.withRefMap(child.typ.refMap))(it)
        TableAggregateByKey(child, expr)
      case "TableKeyByAndAggregate" =>
        val nPartitions = opt(it, int32_literal)
        val bufferSize = int32_literal(it)
        val child = table_ir(env)(it)
        val newEnv = env.withRefMap(child.typ.refMap)
        val expr = ir_value_expr(newEnv)(it)
        val newKey = ir_value_expr(newEnv)(it)
        TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize)
      case "TableGroupWithinPartitions" =>
        val name = identifier(it)
        val n = int32_literal(it)
        val child = table_ir(env)(it)
        TableGroupWithinPartitions(child, name, n)
      case "TableRepartition" =>
        val n = int32_literal(it)
        val strategy = int32_literal(it)
        val child = table_ir(env)(it)
        TableRepartition(child, n, strategy)
      case "TableHead" =>
        val n = int64_literal(it)
        val child = table_ir(env)(it)
        TableHead(child, n)
      case "TableTail" =>
        val n = int64_literal(it)
        val child = table_ir(env)(it)
        TableTail(child, n)
      case "TableJoin" =>
        val joinType = identifier(it)
        val joinKey = int32_literal(it)
        val left = table_ir(env)(it)
        val right = table_ir(env)(it)
        TableJoin(left, right, joinType, joinKey)
      case "TableLeftJoinRightDistinct" =>
        val root = identifier(it)
        val left = table_ir(env)(it)
        val right = table_ir(env)(it)
        TableLeftJoinRightDistinct(left, right, root)
      case "TableIntervalJoin" =>
        val root = identifier(it)
        val product = boolean_literal(it)
        val left = table_ir(env)(it)
        val right = table_ir(env)(it)
        TableIntervalJoin(left, right, root, product)
      case "TableMultiWayZipJoin" =>
        val dataName = string_literal(it)
        val globalsName = string_literal(it)
        val children = table_ir_children(env)(it)
        TableMultiWayZipJoin(children, dataName, globalsName)
      case "TableParallelize" =>
        val nPartitions = opt(it, int32_literal)
        val rowsAndGlobal = ir_value_expr(env)(it)
        TableParallelize(rowsAndGlobal, nPartitions)
      case "TableMapRows" =>
        val child = table_ir(env)(it)
        val newRow = ir_value_expr(env.withRefMap(child.typ.refMap))(it)
        TableMapRows(child, newRow)
      case "TableMapGlobals" =>
        val child = table_ir(env)(it)
        val newRow = ir_value_expr(env.withRefMap(child.typ.refMap))(it)
        TableMapGlobals(child, newRow)
      case "TableRange" =>
        val n = int32_literal(it)
        val nPartitions = opt(it, int32_literal)
        TableRange(n, nPartitions.getOrElse(HailContext.backend.defaultParallelism))
      case "TableUnion" =>
        val children = table_ir_children(env)(it)
        TableUnion(children)
      case "TableOrderBy" =>
        val sortFields = sort_fields(it)
        val child = table_ir(env)(it)
        TableOrderBy(child, sortFields)
      case "TableExplode" =>
        val path = string_literals(it)
        val child = table_ir(env)(it)
        TableExplode(child, path)
      case "CastMatrixToTable" =>
        val entriesField = string_literal(it)
        val colsField = string_literal(it)
        val child = matrix_ir(env)(it)
        CastMatrixToTable(child, entriesField, colsField)
      case "MatrixToTableApply" =>
        val config = string_literal(it)
        val child = matrix_ir(env)(it)
        MatrixToTableApply(child, RelationalFunctions.lookupMatrixToTable(env.ctx, config))
      case "TableToTableApply" =>
        val config = string_literal(it)
        val child = table_ir(env)(it)
        TableToTableApply(child, RelationalFunctions.lookupTableToTable(env.ctx, config))
      case "BlockMatrixToTableApply" =>
        val config = string_literal(it)
        val bm = blockmatrix_ir(env)(it)
        val aux = ir_value_expr(env)(it)
        BlockMatrixToTableApply(bm, aux, RelationalFunctions.lookupBlockMatrixToTable(env.ctx, config))
      case "BlockMatrixToTable" =>
        val child = blockmatrix_ir(env)(it)
        BlockMatrixToTable(child)
      case "TableRename" =>
        val rowK = string_literals(it)
        val rowV = string_literals(it)
        val globalK = string_literals(it)
        val globalV = string_literals(it)
        val child = table_ir(env)(it)
        TableRename(child, rowK.zip(rowV).toMap, globalK.zip(globalV).toMap)
      case "TableFilterIntervals" =>
        val intervals = string_literal(it)
        val keep = boolean_literal(it)
        val child = table_ir(env)(it)
        TableFilterIntervals(child,
          JSONAnnotationImpex.importAnnotation(JsonMethods.parse(intervals),
            TArray(TInterval(child.typ.keyType)),
            padNulls = false).asInstanceOf[IndexedSeq[Interval]],
          keep)
      case "RelationalLetTable" =>
        val name = identifier(it)
        val value = ir_value_expr(env)(it)
        val body = table_ir(env)(it)
        RelationalLetTable(name, value, body)
      case "JavaTable" =>
        val name = identifier(it)
        env.irMap(name).asInstanceOf[TableIR]
    }
  }

  def matrix_ir_children(env: IRParserEnvironment)(it: TokenIterator): Array[MatrixIR] =
    repUntil(it, matrix_ir(env), PunctuationToken(")"))

  def matrix_ir(env: IRParserEnvironment)(it: TokenIterator): MatrixIR = {
    punctuation(it, "(")
    val ir = matrix_ir_1(env)(it)
    punctuation(it, ")")
    ir
  }

  def matrix_ir_1(env: IRParserEnvironment)(it: TokenIterator): MatrixIR = {
    identifier(it) match {
      case "MatrixFilterCols" =>
        val child = matrix_ir(env)(it)
        val pred = ir_value_expr(env.withRefMap(child.typ.refMap))(it)
        MatrixFilterCols(child, pred)
      case "MatrixFilterRows" =>
        val child = matrix_ir(env)(it)
        val pred = ir_value_expr(env.withRefMap(child.typ.refMap))(it)
        MatrixFilterRows(child, pred)
      case "MatrixFilterEntries" =>
        val child = matrix_ir(env)(it)
        val pred = ir_value_expr(env.withRefMap(child.typ.refMap))(it)
        MatrixFilterEntries(child, pred)
      case "MatrixMapCols" =>
        val newKey = opt(it, string_literals)
        val child = matrix_ir(env)(it)
        val newCol = ir_value_expr(env.withRefMap(child.typ.refMap) + ("n_rows" -> TInt64))(it)
        MatrixMapCols(child, newCol, newKey.map(_.toFastIndexedSeq))
      case "MatrixKeyRowsBy" =>
        val key = identifiers(it)
        val isSorted = boolean_literal(it)
        val child = matrix_ir(env)(it)
        MatrixKeyRowsBy(child, key, isSorted)
      case "MatrixMapRows" =>
        val child = matrix_ir(env)(it)
        val newRow = ir_value_expr(env.withRefMap(child.typ.refMap) + ("n_cols" -> TInt32))(it)
        MatrixMapRows(child, newRow)
      case "MatrixMapEntries" =>
        val child = matrix_ir(env)(it)
        val newEntry = ir_value_expr(env.withRefMap(child.typ.refMap))(it)
        MatrixMapEntries(child, newEntry)
      case "MatrixUnionCols" =>
        val joinType = identifier(it)
        val left = matrix_ir(env)(it)
        val right = matrix_ir(env)(it)
        MatrixUnionCols(left, right, joinType)
      case "MatrixMapGlobals" =>
        val child = matrix_ir(env)(it)
        val newGlobals = ir_value_expr(env.withRefMap(child.typ.refMap))(it)
        MatrixMapGlobals(child, newGlobals)
      case "MatrixAggregateColsByKey" =>
        val child = matrix_ir(env)(it)
        val newEnv = env.withRefMap(child.typ.refMap)
        val entryExpr = ir_value_expr(newEnv)(it)
        val colExpr = ir_value_expr(newEnv)(it)
        MatrixAggregateColsByKey(child, entryExpr, colExpr)
      case "MatrixAggregateRowsByKey" =>
        val child = matrix_ir(env)(it)
        val newEnv = env.withRefMap(child.typ.refMap)
        val entryExpr = ir_value_expr(newEnv)(it)
        val rowExpr = ir_value_expr(newEnv)(it)
        MatrixAggregateRowsByKey(child, entryExpr, rowExpr)
      case "MatrixRead" =>
        val requestedType = opt(it, matrix_type_expr(env.typEnv))
        val dropCols = boolean_literal(it)
        val dropRows = boolean_literal(it)
        val readerStr = string_literal(it)
        val reader = MatrixReader.fromJson(env, JsonMethods.parse(readerStr).asInstanceOf[JObject])
        MatrixRead(requestedType.getOrElse(reader.fullMatrixType), dropCols, dropRows, reader)
      case "MatrixAnnotateRowsTable" =>
        val root = string_literal(it)
        val product = boolean_literal(it)
        val child = matrix_ir(env)(it)
        val table = table_ir(env)(it)
        MatrixAnnotateRowsTable(child, table, root, product)
      case "MatrixAnnotateColsTable" =>
        val root = string_literal(it)
        val child = matrix_ir(env)(it)
        val table = table_ir(env)(it)
        MatrixAnnotateColsTable(child, table, root)
      case "MatrixExplodeRows" =>
        val path = identifiers(it)
        val child = matrix_ir(env)(it)
        MatrixExplodeRows(child, path)
      case "MatrixExplodeCols" =>
        val path = identifiers(it)
        val child = matrix_ir(env)(it)
        MatrixExplodeCols(child, path)
      case "MatrixChooseCols" =>
        val oldIndices = int32_literals(it)
        val child = matrix_ir(env)(it)
        MatrixChooseCols(child, oldIndices)
      case "MatrixCollectColsByKey" =>
        val child = matrix_ir(env)(it)
        MatrixCollectColsByKey(child)
      case "MatrixRepartition" =>
        val n = int32_literal(it)
        val strategy = int32_literal(it)
        val child = matrix_ir(env)(it)
        MatrixRepartition(child, n, strategy)
      case "MatrixUnionRows" =>
        val children = matrix_ir_children(env)(it)
        MatrixUnionRows(children)
      case "MatrixDistinctByRow" =>
        val child = matrix_ir(env)(it)
        MatrixDistinctByRow(child)
      case "MatrixRowsHead" =>
        val n = int64_literal(it)
        val child = matrix_ir(env)(it)
        MatrixRowsHead(child, n)
      case "MatrixColsHead" =>
        val n = int32_literal(it)
        val child = matrix_ir(env)(it)
        MatrixColsHead(child, n)
      case "MatrixRowsTail" =>
        val n = int64_literal(it)
        val child = matrix_ir(env)(it)
        MatrixRowsTail(child, n)
      case "MatrixColsTail" =>
        val n = int32_literal(it)
        val child = matrix_ir(env)(it)
        MatrixColsTail(child, n)
      case "CastTableToMatrix" =>
        val entriesField = identifier(it)
        val colsField = identifier(it)
        val colKey = identifiers(it)
        val child = table_ir(env)(it)
        CastTableToMatrix(child, entriesField, colsField, colKey)
      case "MatrixToMatrixApply" =>
        val config = string_literal(it)
        val child = matrix_ir(env)(it)
        MatrixToMatrixApply(child, RelationalFunctions.lookupMatrixToMatrix(env.ctx, config))
      case "MatrixRename" =>
        val globalK = string_literals(it)
        val globalV = string_literals(it)
        val colK = string_literals(it)
        val colV = string_literals(it)
        val rowK = string_literals(it)
        val rowV = string_literals(it)
        val entryK = string_literals(it)
        val entryV = string_literals(it)
        val child = matrix_ir(env)(it)
        MatrixRename(child, globalK.zip(globalV).toMap, colK.zip(colV).toMap, rowK.zip(rowV).toMap, entryK.zip(entryV).toMap)
      case "MatrixFilterIntervals" =>
        val intervals = string_literal(it)
        val keep = boolean_literal(it)
        val child = matrix_ir(env)(it)
        MatrixFilterIntervals(child,
          JSONAnnotationImpex.importAnnotation(JsonMethods.parse(intervals),
            TArray(TInterval(child.typ.rowKeyStruct)),
            padNulls = false).asInstanceOf[IndexedSeq[Interval]],
          keep)
      case "RelationalLetMatrixTable" =>
        val name = identifier(it)
        val value = ir_value_expr(env)(it)
        val body = matrix_ir(env)(it)
        RelationalLetMatrixTable(name, value, body)
      case "JavaMatrix" =>
        val name = identifier(it)
        env.irMap(name).asInstanceOf[MatrixIR]
      case "JavaMatrixVectorRef" =>
        val id = int32_literal(it)
        val idx = int32_literal(it)
        HailContext.get.irVectors(id)(idx).asInstanceOf[MatrixIR]
    }
  }

  def blockmatrix_sparsifier(env: IRParserEnvironment)(it: TokenIterator): BlockMatrixSparsifier = {
    punctuation(it, "(")
    identifier(it) match {
      case "PyRowIntervalSparsifier" =>
        val blocksOnly = boolean_literal(it)
        punctuation(it, ")")
        val Row(starts: IndexedSeq[Long @unchecked], stops: IndexedSeq[Long @unchecked]) =
          ExecuteContext.scoped() { ctx => CompileAndEvaluate[Row](ctx, ir_value_expr(env)(it)) }
        RowIntervalSparsifier(blocksOnly, starts, stops)
      case "PyBandSparsifier" =>
        val blocksOnly = boolean_literal(it)
        punctuation(it, ")")
        val Row(l: Long, u: Long) =
          ExecuteContext.scoped() { ctx => CompileAndEvaluate[Row](ctx, ir_value_expr(env)(it)) }
        BandSparsifier(blocksOnly, l, u)
      case "PyPerBlockSparsifier" =>
        punctuation(it, ")")
        val indices: IndexedSeq[Int] =
          ExecuteContext.scoped() { ctx => CompileAndEvaluate[IndexedSeq[Int]](ctx, ir_value_expr(env)(it)) }
        PerBlockSparsifier(indices)
      case "PyRectangleSparsifier" =>
        punctuation(it, ")")
        val rectangles: IndexedSeq[Long] =
          ExecuteContext.scoped() { ctx => CompileAndEvaluate[IndexedSeq[Long]](ctx, ir_value_expr(env)(it)) }
        RectangleSparsifier(rectangles.grouped(4).toIndexedSeq)
      case "RowIntervalSparsifier" =>
        val blocksOnly = boolean_literal(it)
        val starts = int64_literals(it)
        val stops = int64_literals(it)
        punctuation(it, ")")
        RowIntervalSparsifier(blocksOnly, starts, stops)
      case "BandSparsifier" =>
        val blocksOnly = boolean_literal(it)
        val l = int64_literal(it)
        val u = int64_literal(it)
        punctuation(it, ")")
        BandSparsifier(blocksOnly, l, u)
      case "RectangleSparsifier" =>
        val rectangles = int64_literals(it).toFastIndexedSeq
        punctuation(it, ")")
        RectangleSparsifier(rectangles.grouped(4).toIndexedSeq)
    }
  }

  def blockmatrix_ir(env: IRParserEnvironment)(it: TokenIterator): BlockMatrixIR = {
    punctuation(it, "(")
    val ir = blockmatrix_ir1(env)(it)
    punctuation(it, ")")
    ir
  }

  def blockmatrix_ir1(env: IRParserEnvironment)(it: TokenIterator): BlockMatrixIR = {
    identifier(it) match {
      case "BlockMatrixRead" =>
        val readerStr = string_literal(it)
        val reader = BlockMatrixReader.fromJValue(env.ctx, JsonMethods.parse(readerStr))
        BlockMatrixRead(reader)
      case "BlockMatrixMap" =>
        val name = identifier(it)
        val needs_dense = boolean_literal(it)
        val child = blockmatrix_ir(env)(it)
        val f = ir_value_expr(env + (name -> child.typ.elementType))(it)
        BlockMatrixMap(child, name, f, needs_dense)
      case "BlockMatrixMap2" =>
        val lName = identifier(it)
        val rName = identifier(it)
        val sparsityStrategy = SparsityStrategy.fromString(identifier(it))
        val left = blockmatrix_ir(env)(it)
        val right = blockmatrix_ir(env)(it)
        val f = ir_value_expr(env.update(Map(lName -> left.typ.elementType, rName -> right.typ.elementType)))(it)
        BlockMatrixMap2(left, right, lName, rName, f, sparsityStrategy)
      case "BlockMatrixDot" =>
        val left = blockmatrix_ir(env)(it)
        val right = blockmatrix_ir(env)(it)
        BlockMatrixDot(left, right)
      case "BlockMatrixBroadcast" =>
        val inIndexExpr = int32_literals(it)
        val shape = int64_literals(it)
        val blockSize = int32_literal(it)
        val child = blockmatrix_ir(env)(it)
        BlockMatrixBroadcast(child, inIndexExpr, shape, blockSize)
      case "BlockMatrixAgg" =>
        val outIndexExpr = int32_literals(it)
        val child = blockmatrix_ir(env)(it)
        BlockMatrixAgg(child, outIndexExpr)
      case "BlockMatrixFilter" =>
        val indices = literals(literals(int64_literal))(it)
        val child = blockmatrix_ir(env)(it)
        BlockMatrixFilter(child, indices)
      case "BlockMatrixDensify" =>
        val child = blockmatrix_ir(env)(it)
        BlockMatrixDensify(child)
      case "BlockMatrixSparsify" =>
        val sparsifier = blockmatrix_sparsifier(env)(it)
        val child = blockmatrix_ir(env)(it)
        BlockMatrixSparsify(child, sparsifier)
      case "BlockMatrixSlice" =>
        val slices = literals(literals(int64_literal))(it)
        val child = blockmatrix_ir(env)(it)
        BlockMatrixSlice(child, slices.map(_.toFastIndexedSeq).toFastIndexedSeq)
      case "ValueToBlockMatrix" =>
        val shape = int64_literals(it)
        val blockSize = int32_literal(it)
        val child = ir_value_expr(env)(it)
        ValueToBlockMatrix(child, shape, blockSize)
      case "BlockMatrixRandom" =>
        val seed = int64_literal(it)
        val gaussian = boolean_literal(it)
        val shape = int64_literals(it)
        val blockSize = int32_literal(it)
        BlockMatrixRandom(seed, gaussian, shape, blockSize)
      case "RelationalLetBlockMatrix" =>
        val name = identifier(it)
        val value = ir_value_expr(env)(it)
        val body = blockmatrix_ir(env)(it)
        RelationalLetBlockMatrix(name, value, body)
      case "JavaBlockMatrix" =>
        val name = identifier(it)
        env.irMap(name).asInstanceOf[BlockMatrixIR]
    }
  }

  def parse[T](s: String, f: (TokenIterator) => T): T = {
    val it = IRLexer.parse(s).toIterator.buffered
    f(it)
  }

  def parse_value_ir(s: String, env: IRParserEnvironment): IR = parse(s, ir_value_expr(env))

  def parse_value_ir(ctx: ExecuteContext, s: String): IR = {
    parse_value_ir(s, IRParserEnvironment(ctx))
  }

  def parse_table_ir(ctx: ExecuteContext, s: String): TableIR = parse_table_ir(s, IRParserEnvironment(ctx))

  def parse_table_ir(s: String, env: IRParserEnvironment): TableIR = parse(s, table_ir(env))

  def parse_matrix_ir(s: String, env: IRParserEnvironment): MatrixIR = parse(s, matrix_ir(env))

  def parse_matrix_ir(ctx: ExecuteContext, s: String): MatrixIR = parse_matrix_ir(s, IRParserEnvironment(ctx))

  def parse_blockmatrix_ir(s: String, env: IRParserEnvironment): BlockMatrixIR = parse(s, blockmatrix_ir(env))

  def parse_blockmatrix_ir(ctx: ExecuteContext, s: String): BlockMatrixIR = parse_blockmatrix_ir(s, IRParserEnvironment(ctx))

  def parseType(code: String, env: TypeParserEnvironment): Type = parse(code, type_expr(env))

  def parsePType(code: String, env: TypeParserEnvironment): PType = parse(code, ptype_expr(env))

  def parseStructType(code: String, env: TypeParserEnvironment): TStruct = coerce[TStruct](parse(code, type_expr(env)))

  def parseUnionType(code: String, env: TypeParserEnvironment): TUnion = coerce[TUnion](parse(code, type_expr(env)))

  def parseRVDType(code: String, env: TypeParserEnvironment): RVDType = parse(code, rvd_type_expr(env))

  def parseTableType(code: String, env: TypeParserEnvironment): TableType = parse(code, table_type_expr(env))

  def parseMatrixType(code: String, env: TypeParserEnvironment): MatrixType = parse(code, matrix_type_expr(env))

  def parseType(code: String): Type = parseType(code, TypeParserEnvironment.default)

  def parseSortField(code: String): SortField = parse(code, sort_field)

  def parsePType(code: String): PType = parsePType(code, TypeParserEnvironment.default)

  def parseStructType(code: String): TStruct = parseStructType(code, TypeParserEnvironment.default)

  def parseUnionType(code: String): TUnion = parseUnionType(code, TypeParserEnvironment.default)

  def parseRVDType(code: String): RVDType = parseRVDType(code, TypeParserEnvironment.default)

  def parseTableType(code: String): TableType = parseTableType(code, TypeParserEnvironment.default)

  def parseMatrixType(code: String): MatrixType = parseMatrixType(code, TypeParserEnvironment.default)
}

