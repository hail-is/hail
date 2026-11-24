package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.collection.FastSeq
import is.hail.expr.{JSONAnnotationImpex, Nat, ParserUtils}
import is.hail.expr.ir.Scopes.ScopesSyntax
import is.hail.expr.ir.agg._
import is.hail.expr.ir.defs._
import is.hail.expr.ir.functions.RelationalFunctions
import is.hail.expr.ir.lowering.LoweringPipeline
import is.hail.io.{BufferSpec, TypedCodecSpec}
import is.hail.rvd.{RVDPartitioner, RVDType}
import is.hail.types.{tcoerce, VirtualTypeWithReq}
import is.hail.types.encoded.EType
import is.hail.types.physical._
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.utils.StackSafe._
import is.hail.utils.StringEscapeUtils._
import is.hail.variant.{ReferenceGenome, ReferenceGenomeEnv}

import scala.collection.immutable.ArraySeq
import scala.reflect.ClassTag
import scala.util.parsing.input.{OffsetPosition, Positional}

import java.util.Base64

import org.apache.spark.sql.Row
import org.json4s.{Formats, JObject}
import org.json4s.jackson.{JsonMethods, Serialization}

final case class Token(kind: Token.Kind, value: String) extends Positional

object Token {
  final class Kind private[Token] (val name: String) extends AnyVal {
    def apply(value: String): Token = new Token(this, value)
    def unapply(t: Token): Option[String] = Option.when(t.kind == this)(t.value)
    override def toString: String = name
  }

  val Ident = new Kind("identifier")
  val Str = new Kind("string")
  val Num = new Kind("number")
  val Punc = new Kind("punctuation")
}

final class IRLexer(code: String) extends scala.collection.AbstractIterator[Token] {
  import is.hail.expr.ir.{Token => T}

  private[this] var i = 0

  override def hasNext: Boolean = {
    skipWhitespace()
    i < code.length
  }

  override def next(): Token = {
    if (!hasNext) fatal("Unexpected end of input.")

    val start = i

    val token =
      code.charAt(i) match {
        case '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>' | ',' | ':' | '+' | '@' | '=' =>
          i += 1
          T.Punc(code.substring(start, i))
        case delim @ ('"' | '\'') =>
          T.Str(quoted(delim, "string literal"))
        case '`' =>
          T.Ident(quoted('`', "backtick identifier"))
        case c if Character.isJavaIdentifierStart(c) =>
          i += 1
          while (i < code.length && Character.isJavaIdentifierPart(code.charAt(i))) i += 1
          T.Ident(code.substring(start, i))
        case _ =>
          while (
            i < code.length && (Character.isLetterOrDigit(code.charAt(i)) || ".+-".indexOf(
              code.charAt(i).toInt
            ) >= 0)
          ) i += 1

          if (i == start) fail(s"Unexpected character '${code.charAt(start)}'.")
          T.Num(code.substring(start, i))
      }

    token.setPos(OffsetPosition(code, start))
    token
  }

  private def fail(msg: String): Nothing =
    ParserUtils.error(OffsetPosition(code, i), msg)

  private def skipWhitespace(): Unit =
    while (i < code.length && Character.isWhitespace(code.charAt(i))) i += 1

  private def quoted(delim: Char, what: String): String = {
    val start = i + 1
    i = start

    while (i < code.length && code.charAt(i) != delim) {
      if (code.charAt(i) == '\\') i += 1
      i += 1
    }

    if (i >= code.length) fail(s"unterminated $what")
    i += 1
    unescapeString(code.substring(start, i - 1))
  }
}

object IRParser {
  import is.hail.expr.ir.{Token => T}

  private val snd: (Name, Type) => Type = (_, t) => t

  def error(t: Token, msg: String): Nothing = ParserUtils.error(t.pos, msg)

  def expected(what: String, x: Token): Nothing =
    error(x, s"Expected $what but found ${x.kind} '${x.value}'.")

  def deserialize[A](str: String)(implicit formats: Formats, mf: Manifest[A]): A =
    try
      Serialization.read[A](str)
    catch {
      case e: org.json4s.MappingException =>
        throw new RuntimeException(s"Couldn't deserialize $str", e)
    }

  def punctuation(it: TokenIterator, symbol: String): Unit =
    it.next() match {
      case T.Punc(`symbol`) =>
      case x => expected(s"punctuation '$symbol'", x)
    }

  def identifier(it: TokenIterator): String =
    it.next() match {
      case T.Ident(v) => v
      case x => expected("identifier", x)
    }

  def identifier(it: TokenIterator, expectedId: String): Unit =
    it.next() match {
      case T.Ident(`expectedId`) =>
      case x => expected(s"identifier '$expectedId'", x)
    }

  def identifiers(it: TokenIterator): IndexedSeq[String] =
    base_seq_parser(identifier)(it)

  def name(it: TokenIterator): Name = Name(identifier(it))

  def names(it: TokenIterator): IndexedSeq[Name] =
    base_seq_parser(name)(it)

  def boolean_literal(it: TokenIterator): Boolean =
    it.next() match {
      case T.Ident("True") => true
      case T.Ident("False") => false
      case x => expected("boolean", x)
    }

  def int32_literal(it: TokenIterator): Int =
    it.next() match {
      case x @ T.Num(n) => n.toIntOption.getOrElse(expected("int32", x))
      case x => expected("integer", x)
    }

  def int64_literal(it: TokenIterator): Long =
    it.next() match {
      case x @ T.Num(n) => n.toLongOption.getOrElse(expected("int64", x))
      case x => expected("integer", x)
    }

  def float32_literal(it: TokenIterator): Float =
    it.next() match {
      case x @ (T.Num(_) | T.Ident(_)) =>
        x.value.toDoubleOption.map(_.toFloat).getOrElse(expected("float32", x))
      case x =>
        expected("float", x)
    }

  def float64_literal(it: TokenIterator): Double =
    it.next() match {
      case x @ (T.Num(_) | T.Ident(_)) =>
        x.value.toDoubleOption.getOrElse(expected("float64", x))
      case x =>
        expected("float", x)
    }

  def string_literal(it: TokenIterator): String =
    it.next() match {
      case T.Str(v) => v
      case x => expected("string", x)
    }

  def partitioner_literal(ctx: ExecuteContext)(it: TokenIterator): RVDPartitioner = {
    identifier(it, "Partitioner")
    val keyType = type_expr(ctx.references)(it).asInstanceOf[TStruct]
    val vJSON = JsonMethods.parse(string_literal(it))
    val rangeBounds = JSONAnnotationImpex.importAnnotation(vJSON, TArray(TInterval(keyType)))
    new RVDPartitioner(
      keyType,
      rangeBounds.asInstanceOf[IndexedSeq[Interval]],
    )
  }

  def literals[A](
    literalIdentifier: TokenIterator => A
  )(
    it: TokenIterator
  )(implicit tct: ClassTag[A]
  ): IndexedSeq[A] =
    base_seq_parser(literalIdentifier)(it)

  def between[A](
    open: TokenIterator => Any,
    close: TokenIterator => Any,
    f: TokenIterator => A,
  )(
    it: TokenIterator
  ): A = {
    open(it)
    val a = f(it)
    close(it)
    a
  }

  def string_literals: TokenIterator => IndexedSeq[String] = literals(string_literal)
  def int32_literals: TokenIterator => IndexedSeq[Int] = literals(int32_literal)
  def int64_literals: TokenIterator => IndexedSeq[Long] = literals(int64_literal)

  def opt[A](it: TokenIterator, f: (TokenIterator) => A): Option[A] =
    it.head match {
      case T.Ident("None") =>
        it.next(): Unit
        None
      case _ =>
        Some(f(it))
    }

  def repsepUntil[A](
    it: TokenIterator,
    f: (TokenIterator) => A,
    sep: Token,
    end: Token,
  )(implicit tct: ClassTag[A]
  ): IndexedSeq[A] = {
    val xs = ArraySeq.newBuilder[A]
    while (it.head != end) {
      xs += f(it)
      if (it.head == sep)
        it.next(): Unit
    }
    xs.result()
  }

  def repUntil[A](
    it: TokenIterator,
    f: (TokenIterator) => StackFrame[A],
    end: Token,
  )(implicit tct: ClassTag[A]
  ): StackFrame[IndexedSeq[A]] = {
    val xs = ArraySeq.newBuilder[A]
    var cont: A => StackFrame[IndexedSeq[A]] = null
    def loop(): StackFrame[IndexedSeq[A]] =
      if (it.head != end) {
        f(it).flatMap(cont)
      } else {
        done(xs.result())
      }
    cont = { t =>
      xs += t
      loop()
    }
    loop()
  }

  def repUntilNonStackSafe[A](
    it: TokenIterator,
    f: (TokenIterator) => A,
    end: Token,
  )(implicit tct: ClassTag[A]
  ): IndexedSeq[A] = {
    val xs = ArraySeq.newBuilder[A]
    while (it.head != end)
      xs += f(it)
    xs.result()
  }

  def base_seq_parser[A: ClassTag](f: TokenIterator => A)(it: TokenIterator): IndexedSeq[A] = {
    punctuation(it, "(")
    val r = repUntilNonStackSafe(it, f, T.Punc(")"))
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

  def subset_field[A](f: TokenIterator => A)(it: TokenIterator): (Int, A) = {
    val i = int32_literal(it)
    punctuation(it, ":")
    val t = f(it)
    i -> t
  }

  def struct_field[A](f: TokenIterator => A)(it: TokenIterator): (String, A) = {
    val name = identifier(it)
    punctuation(it, ":")
    val typ = f(it)
    while (it.head == T.Punc("@"))
      decorator(it): Unit
    (name, typ)
  }

  def ptype_field(rgs: ReferenceGenomeEnv)(it: TokenIterator): (String, PType) =
    struct_field(ptype_expr(rgs))(it)

  def type_field(rgs: ReferenceGenomeEnv)(it: TokenIterator): (String, Type) =
    struct_field(type_expr(rgs))(it)

  def vtwr_expr(rgs: ReferenceGenomeEnv)(it: TokenIterator): VirtualTypeWithReq = {
    val pt = ptype_expr(rgs)(it)
    VirtualTypeWithReq(pt)
  }

  def reference_genome(rgs: ReferenceGenomeEnv)(it: TokenIterator): ReferenceGenome = {
    punctuation(it, "(")
    val rg = rgs.resolve(identifier(it))
    punctuation(it, ")")
    rg
  }

  def ptype_expr(rgs: ReferenceGenomeEnv)(it: TokenIterator): PType = {
    val req = it.head match {
      case T.Punc("+") =>
        punctuation(it, "+")
        true
      case _ => false
    }

    val typ = identifier(it) match {
      case "PCInterval" =>
        punctuation(it, "[")
        val pointType = ptype_expr(rgs)(it)
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
        val rg = reference_genome(rgs)(it)
        PCanonicalLocus(rg, req)
      case "PPackedLocus" =>
        val rg = reference_genome(rgs)(it)
        PPackedLocus(rg, req)
      case "PCCall" => PCanonicalCall(req)
      case "PCArray" =>
        punctuation(it, "[")
        val elementType = ptype_expr(rgs)(it)
        punctuation(it, "]")
        PCanonicalArray(elementType, req)
      case "PCNDArray" =>
        punctuation(it, "[")
        val elementType = ptype_expr(rgs)(it)
        punctuation(it, ",")
        val nDims = int32_literal(it)
        punctuation(it, "]")
        PCanonicalNDArray(elementType, nDims, req)
      case "PCSet" =>
        punctuation(it, "[")
        val elementType = ptype_expr(rgs)(it)
        punctuation(it, "]")
        PCanonicalSet(elementType, req)
      case "PCDict" =>
        punctuation(it, "[")
        val keyType = ptype_expr(rgs)(it)
        punctuation(it, ",")
        val valueType = ptype_expr(rgs)(it)
        punctuation(it, "]")
        PCanonicalDict(keyType, valueType, req)
      case "PCTuple" =>
        punctuation(it, "[")
        val fields =
          repsepUntil(it, subset_field(ptype_expr(rgs)), T.Punc(","), T.Punc("]"))
        punctuation(it, "]")
        PCanonicalTuple(fields.map { case (idx, t) => PTupleField(idx, t) }, req)
      case "PCStruct" =>
        punctuation(it, "{")
        val args = repsepUntil(it, ptype_field(rgs), T.Punc(","), T.Punc("}"))
        punctuation(it, "}")
        val fields = args.zipWithIndex.map { case ((id, t), i) => PField(id, t, i) }
        PCanonicalStruct(fields, req)
    }
    assert(typ.required == req)
    typ
  }

  def type_exprs(rgs: ReferenceGenomeEnv)(it: TokenIterator): IndexedSeq[Type] =
    base_seq_parser(type_expr(rgs))(it)

  def type_expr(rgs: ReferenceGenomeEnv)(it: TokenIterator): Type = {
    // skip requiredness token for back-compatibility
    it.head match {
      case T.Punc("+") =>
        punctuation(it, "+")
      case _ =>
    }

    val typ = identifier(it) match {
      case "Interval" =>
        punctuation(it, "[")
        val pointType = type_expr(rgs)(it)
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
        val rg = reference_genome(rgs)(it)
        TLocus(rg)
      case "Call" => TCall
      case "Stream" =>
        punctuation(it, "[")
        val elementType = type_expr(rgs)(it)
        punctuation(it, "]")
        TStream(elementType)
      case "Array" =>
        punctuation(it, "[")
        val elementType = type_expr(rgs)(it)
        punctuation(it, "]")
        TArray(elementType)
      case "NDArray" =>
        punctuation(it, "[")
        val elementType = type_expr(rgs)(it)
        punctuation(it, ",")
        val nDims = int32_literal(it)
        punctuation(it, "]")
        TNDArray(elementType, Nat(nDims))
      case "Set" =>
        punctuation(it, "[")
        val elementType = type_expr(rgs)(it)
        punctuation(it, "]")
        TSet(elementType)
      case "Dict" =>
        punctuation(it, "[")
        val keyType = type_expr(rgs)(it)
        punctuation(it, ",")
        val valueType = type_expr(rgs)(it)
        punctuation(it, "]")
        TDict(keyType, valueType)
      case "Tuple" =>
        punctuation(it, "[")
        val types = repsepUntil(it, type_expr(rgs), T.Punc(","), T.Punc("]"))
        punctuation(it, "]")
        TTuple(types: _*)
      case "TupleSubset" =>
        punctuation(it, "[")
        val fields =
          repsepUntil(it, subset_field(type_expr(rgs)), T.Punc(","), T.Punc("]"))
        punctuation(it, "]")
        TTuple(fields.map { case (idx, t) => TupleField(idx, t) })
      case "Struct" =>
        punctuation(it, "{")
        val args = repsepUntil(it, type_field(rgs), T.Punc(","), T.Punc("}"))
        punctuation(it, "}")
        val fields = args.zipWithIndex.map { case ((id, t), i) => Field(id, t, i) }
        TStruct(fields)
      case "Void" => TVoid
    }
    typ
  }

  def sort_fields(it: TokenIterator): IndexedSeq[SortField] =
    base_seq_parser(sort_field)(it)

  def sort_field(it: TokenIterator): SortField = {
    val sortField = identifier(it)
    val field = sortField.substring(1)
    val sortOrder = SortOrder.parse(sortField.substring(0, 1))
    SortField(field, sortOrder)
  }

  def keys(it: TokenIterator): IndexedSeq[String] = {
    punctuation(it, "[")
    val keys = repsepUntil(it, identifier, T.Punc(","), T.Punc("]"))
    punctuation(it, "]")
    keys
  }

  def trailing_keys(it: TokenIterator): IndexedSeq[String] = {
    it.head match {
      case T.Punc("]") =>
        ArraySeq.empty
      case T.Punc(",") =>
        punctuation(it, ",")
        repsepUntil(it, identifier, T.Punc(","), T.Punc("]"))
    }
  }

  def rvd_type_expr(rgs: ReferenceGenomeEnv)(it: TokenIterator): RVDType = {
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
        val rowType = tcoerce[PStruct](ptype_expr(rgs)(it))
        punctuation(it, "}")
        RVDType(rowType, partitionKey ++ restKey)
    }
  }

  def table_type_expr(rgs: ReferenceGenomeEnv)(it: TokenIterator): TableType = {
    identifier(it, "Table")
    punctuation(it, "{")

    identifier(it, "global")
    punctuation(it, ":")
    val globalType = tcoerce[TStruct](type_expr(rgs)(it))
    punctuation(it, ",")

    identifier(it, "key")
    punctuation(it, ":")
    val key = opt(it, keys).getOrElse(ArraySeq.empty[String])
    punctuation(it, ",")

    identifier(it, "row")
    punctuation(it, ":")
    val rowType = tcoerce[TStruct](type_expr(rgs)(it))
    punctuation(it, "}")
    TableType(rowType, key, tcoerce[TStruct](globalType))
  }

  def matrix_type_expr(rgs: ReferenceGenomeEnv)(it: TokenIterator): MatrixType = {
    identifier(it, "Matrix")
    punctuation(it, "{")

    identifier(it, "global")
    punctuation(it, ":")
    val globalType = tcoerce[TStruct](type_expr(rgs)(it))
    punctuation(it, ",")

    identifier(it, "col_key")
    punctuation(it, ":")
    val colKey = keys(it)
    punctuation(it, ",")

    identifier(it, "col")
    punctuation(it, ":")
    val colType = tcoerce[TStruct](type_expr(rgs)(it))
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
    val rowType = tcoerce[TStruct](type_expr(rgs)(it))
    punctuation(it, ",")

    identifier(it, "entry")
    punctuation(it, ":")
    val entryType = tcoerce[TStruct](type_expr(rgs)(it))
    punctuation(it, "}")

    MatrixType(
      tcoerce[TStruct](globalType),
      colKey,
      colType,
      rowPartitionKey ++ rowRestKey,
      rowType,
      entryType,
    )
  }

  def agg_op(it: TokenIterator): AggOp =
    AggOp.fromString(identifier(it))

  def agg_state_signature(ctx: ExecuteContext)(it: TokenIterator): AggStateSig = {
    punctuation(it, "(")
    val sig = identifier(it) match {
      case "TypedStateSig" =>
        val pt = vtwr_expr(ctx.references)(it)
        TypedStateSig(pt)
      case "DownsampleStateSig" =>
        val labelType = vtwr_expr(ctx.references)(it)
        DownsampleStateSig(labelType)
      case "TakeStateSig" =>
        val pt = vtwr_expr(ctx.references)(it)
        TakeStateSig(pt)
      case "ReservoirSampleStateSig" =>
        val pt = vtwr_expr(ctx.references)(it)
        ReservoirSampleStateSig(pt)
      case "DensifyStateSig" =>
        val pt = vtwr_expr(ctx.references)(it)
        DensifyStateSig(pt)
      case "TakeByStateSig" =>
        val vt = vtwr_expr(ctx.references)(it)
        val kt = vtwr_expr(ctx.references)(it)
        TakeByStateSig(vt, kt, Ascending)
      case "CollectStateSig" =>
        val pt = vtwr_expr(ctx.references)(it)
        CollectStateSig(pt)
      case "CollectAsSetStateSig" =>
        val pt = vtwr_expr(ctx.references)(it)
        CollectAsSetStateSig(pt)
      case "CallStatsStateSig" => CallStatsStateSig()
      case "ArrayAggStateSig" =>
        val nested = agg_state_signatures(ctx)(it)
        ArrayAggStateSig(nested)
      case "GroupedStateSig" =>
        val kt = vtwr_expr(ctx.references)(it)
        val nested = agg_state_signatures(ctx)(it)
        GroupedStateSig(kt, nested)
      case "ApproxCDFStateSig" => ApproxCDFStateSig()
      case "FoldStateSig" =>
        val vtwr = vtwr_expr(ctx.references)(it)
        val accumName = name(it)
        val otherAccumName = name(it)
        val combIR = ir_value_expr(
          ctx,
          BindingEnv.eval(accumName -> vtwr.t, otherAccumName -> vtwr.t),
        )(it).run()
        FoldStateSig(vtwr.canonicalEmitType, accumName, otherAccumName, combIR)
    }
    punctuation(it, ")")
    sig
  }

  def agg_state_signatures(ctx: ExecuteContext)(it: TokenIterator): IndexedSeq[AggStateSig] =
    base_seq_parser(agg_state_signature(ctx))(it)

  def p_agg_sigs(ctx: ExecuteContext)(it: TokenIterator): IndexedSeq[PhysicalAggSig] =
    base_seq_parser(p_agg_sig(ctx))(it)

  def p_agg_sig(ctx: ExecuteContext)(it: TokenIterator): PhysicalAggSig = {
    punctuation(it, "(")
    val sig = identifier(it) match {
      case "Grouped" =>
        val pt = vtwr_expr(ctx.references)(it)
        val nested = p_agg_sigs(ctx)(it)
        GroupedAggSig(pt, nested)
      case "ArrayLen" =>
        val knownLength = boolean_literal(it)
        val nested = p_agg_sigs(ctx)(it)
        ArrayLenAggSig(knownLength, nested)
      case "AggElements" =>
        val nested = p_agg_sigs(ctx)(it)
        AggElementsAggSig(nested)
      case op =>
        val state = agg_state_signature(ctx)(it)
        PhysicalAggSig(AggOp.fromString(op), state)
    }
    punctuation(it, ")")
    sig
  }

  def ir_value(rgs: ReferenceGenomeEnv)(it: TokenIterator): (Type, Any) = {
    val typ = type_expr(rgs)(it)
    val s = string_literal(it)
    val vJSON = JsonMethods.parse(s)
    val v = JSONAnnotationImpex.importAnnotation(vJSON, typ)
    (typ, v)
  }

  def named_value_irs(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IndexedSeq[(String, IR)]] =
    repUntil(it, named_value_ir(ctx, env), T.Punc(")"))

  def named_value_ir(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[(String, IR)] = {
    punctuation(it, "(")
    val name = identifier(it)
    ir_value_expr(ctx, env)(it).map { value =>
      punctuation(it, ")")
      (name, value)
    }
  }

  def ir_value_exprs(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IndexedSeq[IR]] = {
    punctuation(it, "(")
    for {
      irs <- ir_value_children(ctx, env)(it)
      _ = punctuation(it, ")")
    } yield irs
  }

  def ir_value_children(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IndexedSeq[IR]] =
    repUntil(it, ir_value_expr(ctx, env), T.Punc(")"))

  def ir_value_expr(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    punctuation(it, "(")
    for {
      ir <- call(ir_value_expr_1(ctx, env)(it))
      _ = punctuation(it, ")")
    } yield ir
  }

  def apply_like(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
    cons: (String, IndexedSeq[Type], IndexedSeq[IR], Type, Int) => IR,
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val errorID = int32_literal(it)
    val function = identifier(it)
    val typeArgs = type_exprs(ctx.references)(it)
    val rt = type_expr(ctx.references)(it)
    ir_value_children(ctx, env)(it).map(args => cons(function, typeArgs, args, rt, errorID))
  }

  def ir_value_expr_1(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] =
    identifier(it) match {
      case "I32" => done(I32(int32_literal(it)))
      case "I64" => done(I64(int64_literal(it)))
      case "F32" => done(F32(float32_literal(it)))
      case "F64" => done(F64(float64_literal(it)))
      case "Str" => done(Str(string_literal(it)))
      case "UUID4" => done(UUID4(identifier(it)))
      case "True" => done(True())
      case "False" => done(False())
      case "Literal" => value_ir_Literal(ctx, env)(it)
      case "EncodedLiteral" => value_ir_EncodedLiteral(ctx, env)(it)
      case "Void" => done(Void())
      case "Cast" => value_ir_Cast(ctx, env)(it)
      case "CastRename" => value_ir_CastRename(ctx, env)(it)
      case "NA" => done(NA(type_expr(ctx.references)(it)))
      case "IsNA" => ir_value_expr(ctx, env)(it).map(IsNA)
      case "Coalesce" => value_ir_Coalesce(ctx, env)(it)
      case "If" => value_ir_If(ctx, env)(it)
      case "Switch" => value_ir_Switch(ctx, env)(it)
      case "Let" | "Block" => value_ir_Block(ctx, env)(it)
      case "AggLet" => value_ir_AggLet(ctx, env)(it)
      case "TailLoop" => value_ir_TailLoop(ctx, env)(it)
      case "Recur" => value_ir_Recur(ctx, env)(it)
      case "Ref" => value_ir_Ref(ctx, env)(it)
      case "ApplyBinaryPrimOp" => value_ir_ApplyBinaryPrimOp(ctx, env)(it)
      case "ApplyUnaryPrimOp" => value_ir_ApplyUnaryPrimOp(ctx, env)(it)
      case "ApplyComparisonOp" => value_ir_ApplyComparisonOp(ctx, env)(it)
      case "MakeArray" => value_ir_MakeArray(ctx, env)(it)
      case "MakeStream" => value_ir_MakeStream(ctx, env)(it)
      case "ArrayRef" => value_ir_ArrayRef(ctx, env)(it)
      case "ArraySlice" => value_ir_ArraySlice(ctx, env)(it)
      case "RNGStateLiteral" => value_ir_RNGStateLiteral(ctx, env)(it)
      case "RNGSplit" => value_ir_RNGSplit(ctx, env)(it)
      case "RNGSplitStatic" => value_ir_RNGSplitStatic(ctx, env)(it)
      case "ArrayLen" => ir_value_expr(ctx, env)(it).map(ArrayLen)
      case "StreamLen" => ir_value_expr(ctx, env)(it).map(StreamLen)
      case "StreamIota" => value_ir_StreamIota(ctx, env)(it)
      case "StreamRange" => value_ir_StreamRange(ctx, env)(it)
      case "StreamGrouped" => value_ir_StreamGrouped(ctx, env)(it)
      case "ArrayZeros" => ir_value_expr(ctx, env)(it).map(ArrayZeros)
      case "ArraySort" => value_ir_ArraySort(ctx, env)(it)
      case "ArrayMaximalIndependentSet" => value_ir_ArrayMaximalIndependentSet(ctx, env)(it)
      case "MakeNDArray" => value_ir_MakeNDArray(ctx, env)(it)
      case "NDArrayShape" => ir_value_expr(ctx, env)(it).map(NDArrayShape)
      case "NDArrayReshape" => value_ir_NDArrayReshape(ctx, env)(it)
      case "NDArrayConcat" => value_ir_NDArrayConcat(ctx, env)(it)
      case "NDArrayMap" => value_ir_NDArrayMap(ctx, env)(it)
      case "NDArrayMap2" => value_ir_NDArrayMap2(ctx, env)(it)
      case "NDArrayReindex" => value_ir_NDArrayReindex(ctx, env)(it)
      case "NDArrayAgg" => value_ir_NDArrayAgg(ctx, env)(it)
      case "NDArrayRef" => value_ir_NDArrayRef(ctx, env)(it)
      case "NDArraySlice" => value_ir_NDArraySlice(ctx, env)(it)
      case "NDArrayFilter" => value_ir_NDArrayFilter(ctx, env)(it)
      case "NDArrayMatMul" => value_ir_NDArrayMatMul(ctx, env)(it)
      case "NDArrayWrite" => value_ir_NDArrayWrite(ctx, env)(it)
      case "NDArrayQR" => value_ir_NDArrayQR(ctx, env)(it)
      case "NDArraySVD" => value_ir_NDArraySVD(ctx, env)(it)
      case "NDArrayEigh" => value_ir_NDArrayEigh(ctx, env)(it)
      case "NDArrayInv" => value_ir_NDArrayInv(ctx, env)(it)
      case "ToSet" => ir_value_expr(ctx, env)(it).map(ToSet)
      case "ToDict" => ir_value_expr(ctx, env)(it).map(ToDict)
      case "ToArray" => ir_value_expr(ctx, env)(it).map(ToArray)
      case "CastToArray" => ir_value_expr(ctx, env)(it).map(CastToArray)
      case "ToStream" => value_ir_ToStream(ctx, env)(it)
      case "LowerBoundOnOrderedCollection" => value_ir_LowerBoundOnOrderedCollection(ctx, env)(it)
      case "GroupByKey" => ir_value_expr(ctx, env)(it).map(GroupByKey)
      case "StreamBufferedAggregate" => value_ir_StreamBufferedAggregate(ctx, env)(it)
      case "StreamMap" => value_ir_StreamMap(ctx, env)(it)
      case "StreamTake" => value_ir_StreamTake(ctx, env)(it)
      case "StreamDrop" => value_ir_StreamDrop(ctx, env)(it)
      case "StreamZip" => value_ir_StreamZip(ctx, env)(it)
      case "StreamZipJoinProducers" => value_ir_StreamZipJoinProducers(ctx, env)(it)
      case "StreamZipJoin" => value_ir_StreamZipJoin(ctx, env)(it)
      case "StreamMultiMerge" => value_ir_StreamMultiMerge(ctx, env)(it)
      case "StreamFilter" => value_ir_StreamFilter(ctx, env)(it)
      case "StreamTakeWhile" => value_ir_StreamTakeWhile(ctx, env)(it)
      case "StreamDropWhile" => value_ir_StreamDropWhile(ctx, env)(it)
      case "StreamFlatMap" => value_ir_StreamFlatMap(ctx, env)(it)
      case "StreamFold" => value_ir_StreamFold(ctx, env)(it)
      case "StreamFold2" => value_ir_StreamFold2(ctx, env)(it)
      case "StreamScan" => value_ir_StreamScan(ctx, env)(it)
      case "StreamWhiten" => value_ir_StreamWhiten(ctx, env)(it)
      case "StreamJoinRightDistinct" => value_ir_StreamJoinRightDistinct(ctx, env)(it)
      case "StreamLeftIntervalJoin" => value_ir_StreamLeftIntervalJoin(ctx, env)(it)
      case "StreamFor" => value_ir_StreamFor(ctx, env)(it)
      case "StreamAgg" => value_ir_StreamAgg(ctx, env)(it)
      case "StreamAggScan" => value_ir_StreamAggScan(ctx, env)(it)
      case "RunAgg" => value_ir_RunAgg(ctx, env)(it)
      case "RunAggScan" => value_ir_RunAggScan(ctx, env)(it)
      case "AggFilter" => value_ir_AggFilter(ctx, env)(it)
      case "AggExplode" => value_ir_AggExplode(ctx, env)(it)
      case "AggGroupBy" => value_ir_AggGroupBy(ctx, env)(it)
      case "AggArrayPerElement" => value_ir_AggArrayPerElement(ctx, env)(it)
      case "ApplyAggOp" => value_ir_ApplyAggOp(ctx, env)(it)
      case "ApplyScanOp" => value_ir_ApplyScanOp(ctx, env)(it)
      case "AggFold" => value_ir_AggFold(ctx, env)(it)
      case "InitOp" => value_ir_InitOp(ctx, env)(it)
      case "SeqOp" => value_ir_SeqOp(ctx, env)(it)
      case "CombOp" => value_ir_CombOp(ctx, env)(it)
      case "ResultOp" => value_ir_ResultOp(ctx, env)(it)
      case "AggStateValue" => value_ir_AggStateValue(ctx, env)(it)
      case "InitFromSerializedValue" => value_ir_InitFromSerializedValue(ctx, env)(it)
      case "CombOpValue" => value_ir_CombOpValue(ctx, env)(it)
      case "SerializeAggs" => value_ir_SerializeAggs(ctx, env)(it)
      case "DeserializeAggs" => value_ir_DeserializeAggs(ctx, env)(it)
      case "Begin" => ir_value_children(ctx, env)(it).map(Begin(_))
      case "MakeStruct" => named_value_irs(ctx, env)(it).map(MakeStruct(_))
      case "SelectFields" => value_ir_SelectFields(ctx, env)(it)
      case "InsertFields" => value_ir_InsertFields(ctx, env)(it)
      case "GetField" => value_ir_GetField(ctx, env)(it)
      case "MakeTuple" => value_ir_MakeTuple(ctx, env)(it)
      case "GetTupleElement" => value_ir_GetTupleElement(ctx, env)(it)
      case "Die" => value_ir_Die(ctx, env)(it)
      case "ConsoleLog" => value_ir_ConsoleLog(ctx, env)(it)
      case "ApplyIR" => value_ir_ApplyIR(ctx, env)(it)
      case "ApplySpecial" => value_ir_ApplySpecial(ctx, env)(it)
      case "Apply" => value_ir_Apply(ctx, env)(it)
      case "MatrixCount" => value_ir_MatrixCount(ctx, env)(it)
      case "TableCount" => value_ir_TableCount(ctx, env)(it)
      case "TableGetGlobals" => value_ir_TableGetGlobals(ctx, env)(it)
      case "TableCollect" => value_ir_TableCollect(ctx, env)(it)
      case "TableAggregate" => value_ir_TableAggregate(ctx, env)(it)
      case "TableToValueApply" => value_ir_TableToValueApply(ctx, env)(it)
      case "MatrixToValueApply" => value_ir_MatrixToValueApply(ctx, env)(it)
      case "BlockMatrixToValueApply" => value_ir_BlockMatrixToValueApply(ctx, env)(it)
      case "BlockMatrixCollect" => value_ir_BlockMatrixCollect(ctx, env)(it)
      case "TableWrite" => value_ir_TableWrite(ctx, env)(it)
      case "TableMultiWrite" => value_ir_TableMultiWrite(ctx, env)(it)
      case "MatrixAggregate" => value_ir_MatrixAggregate(ctx, env)(it)
      case "MatrixWrite" => value_ir_MatrixWrite(ctx, env)(it)
      case "MatrixMultiWrite" => value_ir_MatrixMultiWrite(ctx, env)(it)
      case "BlockMatrixWrite" => value_ir_BlockMatrixWrite(ctx, env)(it)
      case "BlockMatrixMultiWrite" => value_ir_BlockMatrixMultiWrite(ctx, env)(it)
      case "CollectDistributedArray" => value_ir_CollectDistributedArray(ctx, env)(it)
      case "JavaIR" => value_ir_JavaIR(ctx, env)(it)
      case "ReadPartition" => value_ir_ReadPartition(ctx, env)(it)
      case "WritePartition" => value_ir_WritePartition(ctx, env)(it)
      case "WriteMetadata" => value_ir_WriteMetadata(ctx, env)(it)
      case "ReadValue" => value_ir_ReadValue(ctx, env)(it)
      case "WriteValue" => value_ir_WriteValue(ctx, env)(it)
    }

  private def value_ir_Literal(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val (t, v) = ir_value(ctx.references)(it)
    done(Literal.coerce(t, v))
  }

  private def value_ir_EncodedLiteral(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val typ = type_expr(ctx.references)(it)
    val encodedValue = Base64.getDecoder.decode(string_literal(it))
    val codec = TypedCodecSpec(
      EType.fromPythonTypeEncoding(typ),
      typ,
      BufferSpec.unblockedUncompressed,
    )
    done(EncodedLiteral(codec, Array(encodedValue)))
  }

  private def value_ir_Cast(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val typ = type_expr(ctx.references)(it)
    ir_value_expr(ctx, env)(it).map(Cast(_, typ))
  }

  private def value_ir_CastRename(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val typ = type_expr(ctx.references)(it)
    ir_value_expr(ctx, env)(it).map(CastRename(_, typ))
  }

  private def value_ir_Coalesce(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] =
    for {
      children <- ir_value_children(ctx, env)(it)
      _ = require(children.nonEmpty)
    } yield Coalesce(children)

  private def value_ir_If(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] =
    for {
      cond <- ir_value_expr(ctx, env)(it)
      consq <- ir_value_expr(ctx, env)(it)
      altr <- ir_value_expr(ctx, env)(it)
    } yield If(cond, consq, altr)

  private def value_ir_Switch(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] =
    for {
      x <- ir_value_expr(ctx, env)(it)
      default <- ir_value_expr(ctx, env)(it)
      cases <- ir_value_children(ctx, env)(it)
    } yield Switch(x, default, cases)

  private def value_ir_Block(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val names =
      repUntilNonStackSafe(it, it => (identifier(it), name(it)), T.Punc("("))
    val bindings = new Array[Binding](names.length)
    var e = env
    for {
      _ <- names.indices.foldLeft(done(())) { case (update, i) =>
        val (bindType, name) = names(i)
        val scope = bindType match {
          case "eval" => Scope.EVAL
          case "agg" => Scope.AGG
          case "scan" => Scope.SCAN
        }
        for {
          _ <- update
          value <- ir_value_expr(ctx, Binds.aggSeq(e, scope))(it)
        } yield {
          bindings(i) = Binding(name, value, scope)
          e = e.bind(scope, FastSeq(name -> value.typ))
        }
      }
      body <- ir_value_expr(ctx, e)(it)
    } yield Block(ArraySeq.unsafeWrapArray(bindings), body)
  }

  private def value_ir_AggLet(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val n = name(it)
    val isScan = boolean_literal(it)
    val scope = if (isScan) Scope.SCAN else Scope.AGG
    for {
      value <- ir_value_expr(ctx, Binds.aggSeq(env, scope))(it)
      body <- ir_value_expr(ctx, env.bind(scope, FastSeq(n -> value.typ)))(it)
    } yield AggLet(n, value, body, isScan)
  }

  private def value_ir_TailLoop(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val n = name(it)
    val paramNames = names(it)
    val resultType = type_expr(ctx.references)(it)
    for {
      paramIRs <- fillArray(paramNames.length)(ir_value_expr(ctx, env)(it))(ArraySeq)
      params = paramNames.zip(paramIRs)
      bodyEnv =
        ChildEnv.tailLoop(env, paramNames.length, n, paramNames, resultType, paramIRs)(snd)
      body <- ir_value_expr(ctx, bodyEnv)(it)
    } yield TailLoop(n, params, resultType, body)
  }

  private def value_ir_Recur(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val n = name(it)
    val TTuple(IndexedSeq(_, TupleField(_, rt))) = env.eval.lookup(n)
    ir_value_children(ctx, env)(it).map(args => Recur(n, args, rt))
  }

  private def value_ir_Ref(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val id = name(it)
    done(Ref(id, env.eval(id)))
  }

  private def value_ir_ApplyBinaryPrimOp(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val op = BinaryOp.fromString(identifier(it))
    for {
      l <- ir_value_expr(ctx, env)(it)
      r <- ir_value_expr(ctx, env)(it)
    } yield ApplyBinaryPrimOp(op, l, r)
  }

  private def value_ir_ApplyUnaryPrimOp(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val op = UnaryOp.fromString(identifier(it))
    ir_value_expr(ctx, env)(it).map(ApplyUnaryPrimOp(op, _))
  }

  private def value_ir_ApplyComparisonOp(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val opName = identifier(it)
    for {
      l <- ir_value_expr(ctx, env)(it)
      r <- ir_value_expr(ctx, env)(it)
    } yield ApplyComparisonOp(ComparisonOp.fromString(opName), l, r)
  }

  private def value_ir_MakeArray(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val typ = opt(it, type_expr(ctx.references)).map(_.asInstanceOf[TArray]).orNull
    ir_value_children(ctx, env)(it).map(args => MakeArray.unify(ctx, args, typ))
  }

  private def value_ir_MakeStream(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val typ = opt(it, type_expr(ctx.references)).map(_.asInstanceOf[TStream]).orNull
    val requiresMemoryManagementPerElement = boolean_literal(it)
    ir_value_children(ctx, env)(it).map { args =>
      MakeStream(args, typ, requiresMemoryManagementPerElement)
    }
  }

  private def value_ir_ArrayRef(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val errorID = int32_literal(it)
    for {
      a <- ir_value_expr(ctx, env)(it)
      i <- ir_value_expr(ctx, env)(it)
    } yield ArrayRef(a, i, errorID)
  }

  private def value_ir_ArraySlice(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val errorID = int32_literal(it)
    ir_value_children(ctx, env)(it).map {
      case Seq(a, start, step) => ArraySlice(a, start, None, step, errorID)
      case Seq(a, start, stop, step) => ArraySlice(a, start, Some(stop), step, errorID)
    }
  }

  private def value_ir_RNGStateLiteral(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] =
    done(RNGStateLiteral())

  private def value_ir_RNGSplit(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] =
    for {
      state <- ir_value_expr(ctx, env)(it)
      dynBitstring <- ir_value_expr(ctx, env)(it)
    } yield RNGSplit(state, dynBitstring)

  private def value_ir_RNGSplitStatic(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val staticUid = int64_literal(it)
    ir_value_expr(ctx, env)(it) map { RNGSplitStatic(_, staticUid) }
  }

  private def value_ir_StreamIota(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val requiresMemoryManagementPerElement = boolean_literal(it)
    for {
      start <- ir_value_expr(ctx, env)(it)
      step <- ir_value_expr(ctx, env)(it)
    } yield StreamIota(start, step, requiresMemoryManagementPerElement)
  }

  private def value_ir_StreamRange(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val errorID = int32_literal(it)
    val requiresMemoryManagementPerElement = boolean_literal(it)
    for {
      start <- ir_value_expr(ctx, env)(it)
      stop <- ir_value_expr(ctx, env)(it)
      step <- ir_value_expr(ctx, env)(it)
    } yield StreamRange(start, stop, step, requiresMemoryManagementPerElement, errorID)
  }

  private def value_ir_StreamGrouped(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] =
    for {
      s <- ir_value_expr(ctx, env)(it)
      groupSize <- ir_value_expr(ctx, env)(it)
    } yield StreamGrouped(s, groupSize)

  private def value_ir_ArraySort(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val l = name(it)
    val r = name(it)
    for {
      a <- ir_value_expr(ctx, env)(it)
      lessThanEnv = ChildEnv.arraySort(env, 1, l, r, FastSeq(a))(snd)
      lessThan <- ir_value_expr(ctx, lessThanEnv)(it)
    } yield ArraySort(a, l, r, lessThan)
  }

  private def value_ir_ArrayMaximalIndependentSet(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val hasTieBreaker = boolean_literal(it)
    val bindings = if (hasTieBreaker) Some(name(it) -> name(it)) else None
    for {
      edges <- ir_value_expr(ctx, env)(it)
      tieBreaker <- if (hasTieBreaker) {
        val Some((left, right)) = bindings
        val tieBreakerEnv =
          ChildEnv.arrayMaximalIndependentSet(env, 1, bindings, FastSeq(edges))(snd)
        ir_value_expr(ctx, tieBreakerEnv)(it).map(tbf => Some((left, right, tbf)))
      } else {
        done(None)
      }
    } yield ArrayMaximalIndependentSet(edges, tieBreaker)
  }

  private def value_ir_MakeNDArray(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val errorID = int32_literal(it)
    for {
      data <- ir_value_expr(ctx, env)(it)
      shape <- ir_value_expr(ctx, env)(it)
      rowMajor <- ir_value_expr(ctx, env)(it)
    } yield MakeNDArray(data, shape, rowMajor, errorID)
  }

  private def value_ir_NDArrayReshape(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val errorID = int32_literal(it)
    for {
      nd <- ir_value_expr(ctx, env)(it)
      shape <- ir_value_expr(ctx, env)(it)
    } yield NDArrayReshape(nd, shape, errorID)
  }

  private def value_ir_NDArrayConcat(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val axis = int32_literal(it)
    ir_value_expr(ctx, env)(it).map(nds => NDArrayConcat(nds, axis))
  }

  private def value_ir_NDArrayMap(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val n = name(it)
    for {
      nd <- ir_value_expr(ctx, env)(it)
      bodyEnv = ChildEnv.ndArrayMap(env, 1, n, FastSeq(nd))(snd)
      body <- ir_value_expr(ctx, bodyEnv)(it)
    } yield NDArrayMap(nd, n, body)
  }

  private def value_ir_NDArrayMap2(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val errorID = int32_literal(it)
    val lName = name(it)
    val rName = name(it)
    for {
      l <- ir_value_expr(ctx, env)(it)
      r <- ir_value_expr(ctx, env)(it)
      bodyEnv = ChildEnv.ndArrayMap2(env, 2, lName, rName, FastSeq(l, r))(snd)
      body <- ir_value_expr(ctx, bodyEnv)(it)
    } yield NDArrayMap2(l, r, lName, rName, body, errorID)
  }

  private def value_ir_NDArrayReindex(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val indexExpr = int32_literals(it)
    ir_value_expr(ctx, env)(it).map(nd => NDArrayReindex(nd, indexExpr))
  }

  private def value_ir_NDArrayAgg(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val axes = int32_literals(it)
    ir_value_expr(ctx, env)(it).map(nd => NDArrayAgg(nd, axes))
  }

  private def value_ir_NDArrayRef(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val errorID = int32_literal(it)
    for {
      nd <- ir_value_expr(ctx, env)(it)
      idxs <- ir_value_children(ctx, env)(it)
    } yield NDArrayRef(nd, idxs, errorID)
  }

  private def value_ir_NDArraySlice(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] =
    for {
      nd <- ir_value_expr(ctx, env)(it)
      slices <- ir_value_expr(ctx, env)(it)
    } yield NDArraySlice(nd, slices)

  private def value_ir_NDArrayFilter(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] =
    for {
      nd <- ir_value_expr(ctx, env)(it)
      filters <- repUntil(it, ir_value_expr(ctx, env), T.Punc(")"))
    } yield NDArrayFilter(nd, filters)

  private def value_ir_NDArrayMatMul(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val errorID = int32_literal(it)
    for {
      l <- ir_value_expr(ctx, env)(it)
      r <- ir_value_expr(ctx, env)(it)
    } yield NDArrayMatMul(l, r, errorID)
  }

  private def value_ir_NDArrayWrite(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] =
    for {
      nd <- ir_value_expr(ctx, env)(it)
      path <- ir_value_expr(ctx, env)(it)
    } yield NDArrayWrite(nd, path)

  private def value_ir_NDArrayQR(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val errorID = int32_literal(it)
    val mode = string_literal(it)
    ir_value_expr(ctx, env)(it).map(nd => NDArrayQR(nd, mode, errorID))
  }

  private def value_ir_NDArraySVD(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val errorID = int32_literal(it)
    val fullMatrices = boolean_literal(it)
    val computeUV = boolean_literal(it)
    ir_value_expr(ctx, env)(it).map(nd => NDArraySVD(nd, fullMatrices, computeUV, errorID))
  }

  private def value_ir_NDArrayEigh(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val errorID = int32_literal(it)
    val eigvalsOnly = boolean_literal(it)
    ir_value_expr(ctx, env)(it).map(nd => NDArrayEigh(nd, eigvalsOnly, errorID))
  }

  private def value_ir_NDArrayInv(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val errorID = int32_literal(it)
    ir_value_expr(ctx, env)(it).map(nd => NDArrayInv(nd, errorID))
  }

  private def value_ir_ToStream(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val requiresMemoryManagementPerElement = boolean_literal(it)
    ir_value_expr(ctx, env)(it).map(a => ToStream(a, requiresMemoryManagementPerElement))
  }

  private def value_ir_LowerBoundOnOrderedCollection(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val onKey = boolean_literal(it)
    for {
      col <- ir_value_expr(ctx, env)(it)
      elem <- ir_value_expr(ctx, env)(it)
    } yield LowerBoundOnOrderedCollection(col, elem, onKey)
  }

  private def value_ir_StreamBufferedAggregate(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val n = name(it)
    val aggSigs = p_agg_sigs(ctx)(it)
    val size = int32_literal(it)
    def childEnv(i: Int, cs: IndexedSeq[BaseIR]): BindingEnv[Type] =
      ChildEnv.streamBufferedAggregate(env, i, n, cs)(snd)
    for {
      stream <- ir_value_expr(ctx, env)(it)
      init <- ir_value_expr(ctx, env)(it)
      key <- ir_value_expr(ctx, childEnv(2, FastSeq(stream, init)))(it)
      seq <- ir_value_expr(ctx, childEnv(3, FastSeq(stream, init, key)))(it)
    } yield StreamBufferedAggregate(stream, init, key, seq, n, aggSigs, size)
  }

  private def value_ir_StreamMap(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val n = name(it)
    for {
      a <- ir_value_expr(ctx, env)(it)
      bodyEnv = ChildEnv.streamMap(env, 1, n, FastSeq(a))(snd)
      body <- ir_value_expr(ctx, bodyEnv)(it)
    } yield StreamMap(a, n, body)
  }

  private def value_ir_StreamTake(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] =
    for {
      a <- ir_value_expr(ctx, env)(it)
      num <- ir_value_expr(ctx, env)(it)
    } yield StreamTake(a, num)

  private def value_ir_StreamDrop(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] =
    for {
      a <- ir_value_expr(ctx, env)(it)
      num <- ir_value_expr(ctx, env)(it)
    } yield StreamDrop(a, num)

  private def value_ir_StreamZip(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val errorID = int32_literal(it)
    val behavior = identifier(it) match {
      case "AssertSameLength" => ArrayZipBehavior.AssertSameLength
      case "TakeMinLength" => ArrayZipBehavior.TakeMinLength
      case "ExtendNA" => ArrayZipBehavior.ExtendNA
      case "AssumeSameLength" => ArrayZipBehavior.AssumeSameLength
    }
    val ns = names(it)
    for {
      as <- ns.mapRecur(_ => ir_value_expr(ctx, env)(it))
      bodyEnv = ChildEnv.streamZip(env, ns.length, ns, as)(snd)
      body <- ir_value_expr(ctx, bodyEnv)(it)
    } yield StreamZip(as, ns, body, behavior, errorID)
  }

  private def value_ir_StreamZipJoinProducers(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val key = identifiers(it)
    val ctxName = name(it)
    val curKey = name(it)
    val curVals = name(it)
    def childEnv(i: Int, cs: IndexedSeq[BaseIR]): BindingEnv[Type] =
      ChildEnv.streamZipJoinProducers(env, i, ctxName, key, curKey, curVals, cs)(snd)
    for {
      ctxs <- ir_value_expr(ctx, env)(it)
      makeProducer <- ir_value_expr(ctx, childEnv(1, FastSeq(ctxs)))(it)
      body <- ir_value_expr(ctx, childEnv(2, FastSeq(ctxs, makeProducer)))(it)
    } yield StreamZipJoinProducers(ctxs, ctxName, makeProducer, key, curKey, curVals, body)
  }

  private def value_ir_StreamZipJoin(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val nStreams = int32_literal(it)
    val key = identifiers(it)
    val curKey = name(it)
    val curVals = name(it)
    for {
      streams <- (0 until nStreams).mapRecur(_ => ir_value_expr(ctx, env)(it))
      bodyEnv =
        ChildEnv.streamZipJoin(env, nStreams, nStreams, key, curKey, curVals, streams)(snd)
      body <- ir_value_expr(ctx, bodyEnv)(it)
    } yield StreamZipJoin(streams, key, curKey, curVals, body)
  }

  private def value_ir_StreamMultiMerge(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val key = identifiers(it)
    for {
      streams <- ir_value_exprs(ctx, env)(it)
    } yield StreamMultiMerge(streams, key)
  }

  private def value_ir_StreamFilter(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val n = name(it)
    for {
      a <- ir_value_expr(ctx, env)(it)
      bodyEnv = ChildEnv.streamFilter(env, 1, n, FastSeq(a))(snd)
      body <- ir_value_expr(ctx, bodyEnv)(it)
    } yield StreamFilter(a, n, body)
  }

  private def value_ir_StreamTakeWhile(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val n = name(it)
    for {
      a <- ir_value_expr(ctx, env)(it)
      bodyEnv = ChildEnv.streamTakeWhile(env, 1, n, FastSeq(a))(snd)
      body <- ir_value_expr(ctx, bodyEnv)(it)
    } yield StreamTakeWhile(a, n, body)
  }

  private def value_ir_StreamDropWhile(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val n = name(it)
    for {
      a <- ir_value_expr(ctx, env)(it)
      bodyEnv = ChildEnv.streamDropWhile(env, 1, n, FastSeq(a))(snd)
      body <- ir_value_expr(ctx, bodyEnv)(it)
    } yield StreamDropWhile(a, n, body)
  }

  private def value_ir_StreamFlatMap(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val n = name(it)
    for {
      a <- ir_value_expr(ctx, env)(it)
      bodyEnv = ChildEnv.streamFlatMap(env, 1, n, FastSeq(a))(snd)
      body <- ir_value_expr(ctx, bodyEnv)(it)
    } yield StreamFlatMap(a, n, body)
  }

  private def value_ir_StreamFold(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val accumName = name(it)
    val valueName = name(it)
    for {
      a <- ir_value_expr(ctx, env)(it)
      zero <- ir_value_expr(ctx, env)(it)
      bodyEnv = ChildEnv.streamFold(env, 2, accumName, valueName, FastSeq(a, zero))(snd)
      body <- ir_value_expr(ctx, bodyEnv)(it)
    } yield StreamFold(a, zero, accumName, valueName, body)
  }

  private def value_ir_StreamFold2(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val accumNames = names(it)
    val valueName = name(it)
    val n = accumNames.length
    def childEnv(i: Int, cs: IndexedSeq[BaseIR]): BindingEnv[Type] =
      ChildEnv.streamFold2(env, i, accumNames, valueName, cs)(snd)
    for {
      a <- ir_value_expr(ctx, env)(it)
      accIRs <- fillArray(n)(ir_value_expr(ctx, env)(it))(ArraySeq)
      accs = accumNames.zip(accIRs)
      seqEnv = childEnv(n + 1, a +: accIRs)
      seqs <- fillArray[IR, ArraySeq[IR]](n)(ir_value_expr(ctx, seqEnv)(it))(ArraySeq)
      res <- ir_value_expr(ctx, childEnv(2 * n + 1, a +: accIRs))(it)
    } yield StreamFold2(a, accs, valueName, seqs, res)
  }

  private def value_ir_StreamScan(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val accumName = name(it)
    val valueName = name(it)
    for {
      a <- ir_value_expr(ctx, env)(it)
      zero <- ir_value_expr(ctx, env)(it)
      bodyEnv = ChildEnv.streamScan(env, 2, accumName, valueName, FastSeq(a, zero))(snd)
      body <- ir_value_expr(ctx, bodyEnv)(it)
    } yield StreamScan(a, zero, accumName, valueName, body)
  }

  private def value_ir_StreamWhiten(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val newChunk = identifier(it)
    val prevWindow = identifier(it)
    val vecSize = int32_literal(it)
    val windowSize = int32_literal(it)
    val chunkSize = int32_literal(it)
    val blockSize = int32_literal(it)
    val normalizeAfterWhitening = boolean_literal(it)
    for {
      stream <- ir_value_expr(ctx, env)(it)
    } yield StreamWhiten(stream, newChunk, prevWindow, vecSize, windowSize, chunkSize,
      blockSize, normalizeAfterWhitening)
  }

  private def value_ir_StreamJoinRightDistinct(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val lKey = identifiers(it)
    val rKey = identifiers(it)
    val l = name(it)
    val r = name(it)
    val joinType = identifier(it)
    for {
      left <- ir_value_expr(ctx, env)(it)
      right <- ir_value_expr(ctx, env)(it)
      joinEnv =
        ChildEnv.streamJoinRightDistinct(env, 2, l, r, FastSeq(left, right))(snd)
      join <- ir_value_expr(ctx, joinEnv)(it)
    } yield StreamJoinRightDistinct(left, right, lKey, rKey, l, r, join, joinType)
  }

  private def value_ir_StreamLeftIntervalJoin(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val lKeyFieldName = identifier(it)
    val rIntervalName = identifier(it)
    val lname = name(it)
    val rname = name(it)
    for {
      left <- ir_value_expr(ctx, env)(it)
      right <- ir_value_expr(ctx, env)(it)
      bodyEnv =
        ChildEnv.streamLeftIntervalJoin(env, 2, lname, rname, FastSeq(left, right))(snd)
      body <- ir_value_expr(ctx, bodyEnv)(it)
    } yield StreamLeftIntervalJoin(left, right, lKeyFieldName, rIntervalName, lname, rname,
      body)

  }

  private def value_ir_StreamFor(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val n = name(it)
    for {
      a <- ir_value_expr(ctx, env)(it)
      bodyEnv = ChildEnv.streamFor(env, 1, n, FastSeq(a))(snd)
      body <- ir_value_expr(ctx, bodyEnv)(it)
    } yield StreamFor(a, n, body)
  }

  private def value_ir_StreamAgg(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val n = name(it)
    for {
      a <- ir_value_expr(ctx, env)(it)
      queryEnv = ChildEnv.streamAgg(env, 1, n, FastSeq(a))(snd)
      query <- ir_value_expr(ctx, queryEnv)(it)
    } yield StreamAgg(a, n, query)
  }

  private def value_ir_StreamAggScan(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val n = name(it)
    for {
      a <- ir_value_expr(ctx, env)(it)
      queryEnv = ChildEnv.streamAggScan(env, 1, n, FastSeq(a))(snd)
      query <- ir_value_expr(ctx, queryEnv)(it)
    } yield StreamAggScan(a, n, query)
  }

  private def value_ir_RunAgg(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val signatures = agg_state_signatures(ctx)(it)
    for {
      body <- ir_value_expr(ctx, env)(it)
      result <- ir_value_expr(ctx, env)(it)
    } yield RunAgg(body, result, signatures)
  }

  private def value_ir_RunAggScan(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val n = name(it)
    val signatures = agg_state_signatures(ctx)(it)
    def childEnv(i: Int, cs: IndexedSeq[BaseIR]): BindingEnv[Type] =
      ChildEnv.runAggScan(env, i, n, cs)(snd)
    for {
      array <- ir_value_expr(ctx, env)(it)
      init <- ir_value_expr(ctx, env)(it)
      seq <- ir_value_expr(ctx, childEnv(2, FastSeq(array, init)))(it)
      result <- ir_value_expr(ctx, childEnv(3, FastSeq(array, init, seq)))(it)
    } yield RunAggScan(array, n, init, seq, result, signatures)
  }

  private def value_ir_AggFilter(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val isScan = boolean_literal(it)
    val condEnv = ChildEnv.aggFilter(env, 0, isScan)
    for {
      cond <- ir_value_expr(ctx, condEnv)(it)
      aggIR <- ir_value_expr(ctx, env)(it)
    } yield AggFilter(cond, aggIR, isScan)
  }

  private def value_ir_AggExplode(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val n = name(it)
    val isScan = boolean_literal(it)
    def childEnv(i: Int, cs: IndexedSeq[BaseIR]): BindingEnv[Type] =
      ChildEnv.aggExplode(env, i, n, isScan, cs)(snd)
    for {
      a <- ir_value_expr(ctx, childEnv(0, FastSeq()))(it)
      aggBody <- ir_value_expr(ctx, childEnv(1, FastSeq(a)))(it)
    } yield AggExplode(a, n, aggBody, isScan)
  }

  private def value_ir_AggGroupBy(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val isScan = boolean_literal(it)
    val keyEnv = ChildEnv.aggGroupBy(env, 0, isScan)
    for {
      key <- ir_value_expr(ctx, keyEnv)(it)
      aggIR <- ir_value_expr(ctx, env)(it)
    } yield AggGroupBy(key, aggIR, isScan)
  }

  private def value_ir_AggArrayPerElement(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val elementName = name(it)
    val indexName = name(it)
    val isScan = boolean_literal(it)
    val hasKnownLength = boolean_literal(it)
    def childEnv(i: Int, cs: IndexedSeq[BaseIR]): BindingEnv[Type] =
      ChildEnv.aggArrayPerElement(env, i, elementName, indexName, isScan, cs)(snd)
    for {
      a <- ir_value_expr(ctx, childEnv(0, FastSeq()))(it)
      aggBody <- ir_value_expr(ctx, childEnv(1, FastSeq(a)))(it)
      knownLength <-
        if (hasKnownLength)
          ir_value_expr(ctx, childEnv(2, FastSeq(a, aggBody)))(it).map(Some(_))
        else done(None)
    } yield AggArrayPerElement(a, elementName, indexName, aggBody, knownLength, isScan)
  }

  private def value_ir_ApplyAggOp(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val aggOp = agg_op(it)
    for {
      initOpArgs <- ir_value_exprs(ctx, Binds.aggInit(env, Scope.AGG))(it)
      seqOpArgs <- ir_value_exprs(ctx, Binds.aggSeq(env, Scope.AGG))(it)
    } yield ApplyAggOp(initOpArgs, seqOpArgs, aggOp)
  }

  private def value_ir_ApplyScanOp(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val aggOp = agg_op(it)
    for {
      initOpArgs <- ir_value_exprs(ctx, Binds.aggInit(env, Scope.SCAN))(it)
      seqOpArgs <- ir_value_exprs(ctx, Binds.aggSeq(env, Scope.SCAN))(it)
    } yield ApplyScanOp(initOpArgs, seqOpArgs, aggOp)
  }

  private def value_ir_AggFold(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val accumName = name(it)
    val otherAccumName = name(it)
    val isScan = boolean_literal(it)
    def childEnv(i: Int, cs: IndexedSeq[BaseIR]): BindingEnv[Type] =
      ChildEnv.aggFold(env, i, accumName, otherAccumName, isScan, cs)(snd)
    for {
      zero <- ir_value_expr(ctx, childEnv(0, FastSeq()))(it)
      seqOp <- ir_value_expr(ctx, childEnv(1, FastSeq(zero)))(it)
      combOp <- ir_value_expr(ctx, childEnv(2, FastSeq(zero, seqOp)))(it)
    } yield AggFold(zero, seqOp, combOp, accumName, otherAccumName, isScan)
  }

  private def value_ir_InitOp(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val i = int32_literal(it)
    val aggSig = p_agg_sig(ctx)(it)
    ir_value_exprs(ctx, env)(it).map(args => InitOp(i, args, aggSig))
  }

  private def value_ir_SeqOp(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val i = int32_literal(it)
    val aggSig = p_agg_sig(ctx)(it)
    ir_value_exprs(ctx, env)(it).map(args => SeqOp(i, args, aggSig))
  }

  private def value_ir_CombOp(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val i1 = int32_literal(it)
    val i2 = int32_literal(it)
    val aggSig = p_agg_sig(ctx)(it)
    done(CombOp(i1, i2, aggSig))
  }

  private def value_ir_ResultOp(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val i = int32_literal(it)
    val aggSig = p_agg_sig(ctx)(it)
    done(ResultOp(i, aggSig))
  }

  private def value_ir_AggStateValue(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val i = int32_literal(it)
    val sig = agg_state_signature(ctx)(it)
    done(AggStateValue(i, sig))
  }

  private def value_ir_InitFromSerializedValue(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val i = int32_literal(it)
    val sig = agg_state_signature(ctx)(it)
    ir_value_expr(ctx, env)(it).map(value => InitFromSerializedValue(i, value, sig))
  }

  private def value_ir_CombOpValue(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val i = int32_literal(it)
    val sig = p_agg_sig(ctx)(it)
    ir_value_expr(ctx, env)(it).map(value => CombOpValue(i, value, sig))
  }

  private def value_ir_SerializeAggs(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val i = int32_literal(it)
    val i2 = int32_literal(it)
    val spec = BufferSpec.parse(string_literal(it))
    val aggSigs = agg_state_signatures(ctx)(it)
    done(SerializeAggs(i, i2, spec, aggSigs))
  }

  private def value_ir_DeserializeAggs(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val i = int32_literal(it)
    val i2 = int32_literal(it)
    val spec = BufferSpec.parse(string_literal(it))
    val aggSigs = agg_state_signatures(ctx)(it)
    done(DeserializeAggs(i, i2, spec, aggSigs))
  }

  private def value_ir_SelectFields(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val fields = identifiers(it)
    ir_value_expr(ctx, env)(it).map(old => SelectFields(old, fields))
  }

  private def value_ir_InsertFields(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] =
    for {
      old <- ir_value_expr(ctx, env)(it)
      fieldOrder = opt(it, string_literals)
      fields <- named_value_irs(ctx, env)(it)
    } yield InsertFields(old, fields, fieldOrder)

  private def value_ir_GetField(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val name = identifier(it)
    ir_value_expr(ctx, env)(it).map(s => GetField(s, name))
  }

  private def value_ir_MakeTuple(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val indices = int32_literals(it)
    ir_value_children(ctx, env)(it).map(args => MakeTuple(indices.zip(args)))
  }

  private def value_ir_GetTupleElement(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val idx = int32_literal(it)
    ir_value_expr(ctx, env)(it).map(tuple => GetTupleElement(tuple, idx))
  }

  private def value_ir_Die(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val typ = type_expr(ctx.references)(it)
    val errorID = int32_literal(it)
    ir_value_expr(ctx, env)(it).map(msg => Die(msg, typ, errorID))
  }

  private def value_ir_ConsoleLog(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] =
    for {
      msg <- ir_value_expr(ctx, env)(it)
      result <- ir_value_expr(ctx, env)(it)
    } yield ConsoleLog(msg, result)

  private def value_ir_ApplyIR(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] =
    apply_like(ctx, env, ApplyIR.apply)(it)

  private def value_ir_ApplySpecial(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] =
    apply_like(ctx, env, ApplySpecial)(it)

  private def value_ir_Apply(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    apply_like(
      ctx,
      env,
      (function, typeArgs, args, rt, errorID) =>
        invoke(function, rt, typeArgs, errorID, args: _*),
    )(it)
  }

  private def value_ir_MatrixCount(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] =
    matrix_ir(ctx)(it).map(MatrixCount)

  private def value_ir_TableCount(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] =
    table_ir(ctx)(it).map(TableCount)

  private def value_ir_TableGetGlobals(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] =
    table_ir(ctx)(it).map(TableGetGlobals)

  private def value_ir_TableCollect(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] =
    table_ir(ctx)(it).map(TableCollect)

  private def value_ir_TableAggregate(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] =
    for {
      child <- table_ir(ctx)(it)
      queryEnv = ChildEnv.tableAggregate(env, 1, FastSeq(child))(snd)
      query <- ir_value_expr(ctx, queryEnv)(it)
    } yield TableAggregate(child, query)

  private def value_ir_TableToValueApply(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val config = string_literal(it)
    table_ir(ctx)(it).map { child =>
      TableToValueApply(child, RelationalFunctions.lookupTableToValue(ctx, config))
    }
  }

  private def value_ir_MatrixToValueApply(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val config = string_literal(it)
    matrix_ir(ctx)(it).map { child =>
      MatrixToValueApply(child, RelationalFunctions.lookupMatrixToValue(ctx, config))
    }
  }

  private def value_ir_BlockMatrixToValueApply(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val config = string_literal(it)
    blockmatrix_ir(ctx)(it).map { child =>
      BlockMatrixToValueApply(
        child,
        RelationalFunctions.lookupBlockMatrixToValue(ctx, config),
      )
    }
  }

  private def value_ir_BlockMatrixCollect(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] =
    blockmatrix_ir(ctx)(it).map(BlockMatrixCollect)

  private def value_ir_TableWrite(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    implicit val formats = TableWriter.formats
    val writerStr = string_literal(it)
    table_ir(ctx)(it).map(child => TableWrite(child, deserialize[TableWriter](writerStr)))
  }

  private def value_ir_TableMultiWrite(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    implicit val formats = WrappedMatrixNativeMultiWriter.formats
    val writerStr = string_literal(it)
    table_ir_children(ctx)(it).map { children =>
      TableMultiWrite(children, deserialize[WrappedMatrixNativeMultiWriter](writerStr))
    }
  }

  private def value_ir_MatrixAggregate(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] =
    for {
      child <- matrix_ir(ctx)(it)
      queryEnv = ChildEnv.matrixAggregate(env, 1, FastSeq(child))(snd)
      query <- ir_value_expr(ctx, queryEnv)(it)
    } yield MatrixAggregate(child, query)

  private def value_ir_MatrixWrite(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val writerStr = string_literal(it)
    implicit val formats: Formats = MatrixWriter.formats
    val writer = deserialize[MatrixWriter](writerStr)
    matrix_ir(ctx)(it).map(child => MatrixWrite(child, writer))
  }

  private def value_ir_MatrixMultiWrite(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val writerStr = string_literal(it)
    implicit val formats = MatrixNativeMultiWriter.formats
    val writer = deserialize[MatrixNativeMultiWriter](writerStr)
    matrix_ir_children(ctx)(it).map(children => MatrixMultiWrite(children, writer))
  }

  private def value_ir_BlockMatrixWrite(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val writerStr = string_literal(it)
    implicit val formats: Formats = BlockMatrixWriter.formats
    val writer = deserialize[BlockMatrixWriter](writerStr)
    blockmatrix_ir(ctx)(it).map(child => BlockMatrixWrite(child, writer))
  }

  private def value_ir_BlockMatrixMultiWrite(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val writerStr = string_literal(it)
    implicit val formats: Formats = BlockMatrixWriter.formats
    val writer = deserialize[BlockMatrixMultiWriter](writerStr)
    repUntil(it, blockmatrix_ir(ctx), T.Punc(")")).map { blockMatrices =>
      BlockMatrixMultiWrite(blockMatrices, writer)
    }
  }

  private def value_ir_CollectDistributedArray(
    ctx: ExecuteContext,
    env: BindingEnv[Type],
  )(
    it: TokenIterator
  ): StackFrame[IR] = {
    val staticID = identifier(it)
    val cname = name(it)
    val gname = name(it)
    for {
      ctxs <- ir_value_expr(ctx, env)(it)
      globals <- ir_value_expr(ctx, env)(it)
      bodyEnv =
        ChildEnv.collectDistributedArray(env, 2, cname, gname, FastSeq(ctxs, globals))(snd)
      body <- ir_value_expr(ctx, bodyEnv)(it)
      dynamicID <- ir_value_expr(ctx, env)(it)
    } yield CollectDistributedArray(ctxs, globals, cname, gname, body, dynamicID, staticID)
  }

  private def value_ir_JavaIR(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val id = int32_literal(it)
    done(ctx.PersistedIrCache(id).deepCopy.asInstanceOf[IR])
  }

  private def value_ir_ReadPartition(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    val requestedTypeRaw = it.head match {
      case T.Ident(v @ ("None" | "DropRowUIDs")) =>
        it.next(): Unit
        Left(v)
      case _ =>
        Right(type_expr(ctx.references)(it))
    }
    val reader = PartitionReader.extract(ctx, JsonMethods.parse(string_literal(it)))
    ir_value_expr(ctx, env)(it).map { context =>
      ReadPartition(
        context,
        requestedTypeRaw match {
          case Left("None") => reader.fullRowType
          case Left("DropRowUIDs") => reader.fullRowType.deleteKey(reader.uidFieldName)
          case Right(t) => t.asInstanceOf[TStruct]
        },
        reader,
      )
    }
  }

  private def value_ir_WritePartition(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    implicit val formats: Formats = PartitionWriter.formats(ctx.references)
    val writer = JsonMethods.parse(string_literal(it)).extract[PartitionWriter]
    for {
      stream <- ir_value_expr(ctx, env)(it)
      ctx <- ir_value_expr(ctx, env)(it)
    } yield WritePartition(stream, ctx, writer)
  }

  private def value_ir_WriteMetadata(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    implicit val formats: Formats = MetadataWriter.formats(ctx.references)
    val writer = JsonMethods.parse(string_literal(it)).extract[MetadataWriter]
    ir_value_expr(ctx, env)(it).map(ctx => WriteMetadata(ctx, writer))
  }

  private def value_ir_ReadValue(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    implicit val formats: Formats = ValueReader.formats(ctx.references)
    val reader = JsonMethods.parse(string_literal(it)).extract[ValueReader]
    val typ = type_expr(ctx.references)(it)
    ir_value_expr(ctx, env)(it).map(path => ReadValue(path, reader, typ))
  }

  private def value_ir_WriteValue(ctx: ExecuteContext, env: BindingEnv[Type])(it: TokenIterator)
    : StackFrame[IR] = {
    implicit val formats: Formats = ValueWriter.formats(ctx.references)
    val writer = JsonMethods.parse(string_literal(it)).extract[ValueWriter]
    ir_value_children(ctx, env)(it).map { case Seq(value, path) =>
      WriteValue(value, path, writer)
    }
  }

  def table_ir_children(ctx: ExecuteContext)(it: TokenIterator): StackFrame[IndexedSeq[TableIR]] =
    repUntil(it, table_ir(ctx), T.Punc(")"))

  def table_ir(ctx: ExecuteContext)(it: TokenIterator): StackFrame[TableIR] = {
    punctuation(it, "(")
    for {
      ir <- call(table_ir_1(ctx)(it))
      _ = punctuation(it, ")")
    } yield ir
  }

  def table_ir_1(ctx: ExecuteContext)(it: TokenIterator): StackFrame[TableIR] = {
    identifier(it) match {
      case "TableKeyBy" =>
        val keys = identifiers(it)
        val isSorted = boolean_literal(it)
        val nPartitions = opt(it, int32_literal)
        table_ir(ctx)(it).map(child => TableKeyBy(child, keys, isSorted, nPartitions))
      case "TableDistinct" => table_ir(ctx)(it).map(TableDistinct)
      case "TableFilter" =>
        for {
          child <- table_ir(ctx)(it)
          predEnv = ChildEnv.tableFilter(BindingEnv.empty[Type], 1, FastSeq(child))(snd)
          pred <- ir_value_expr(ctx, predEnv)(it)
        } yield TableFilter(child, pred)
      case "TableRead" =>
        val requestedTypeRaw = it.head match {
          case T.Ident(v @ ("None" | "DropRowUIDs")) =>
            it.next(): Unit
            Left(v)
          case _ =>
            Right(table_type_expr(ctx.references)(it))
        }
        val dropRows = boolean_literal(it)
        val readerStr = string_literal(it)
        val reader =
          TableReader.fromJValue(
            ctx.fs,
            ctx.references,
            JsonMethods.parse(readerStr).asInstanceOf[JObject],
          )
        val requestedType = requestedTypeRaw match {
          case Left("None") => reader.fullType
          case Left("DropRowUIDs") =>
            reader.asInstanceOf[TableReaderWithExtraUID].fullTypeWithoutUIDs
          case Right(t) => t
        }
        done(TableRead(requestedType, dropRows, reader))
      case "MatrixColsTable" => matrix_ir(ctx)(it).map(MatrixColsTable)
      case "MatrixRowsTable" => matrix_ir(ctx)(it).map(MatrixRowsTable)
      case "MatrixEntriesTable" => matrix_ir(ctx)(it).map(MatrixEntriesTable)
      case "TableAggregateByKey" =>
        for {
          child <- table_ir(ctx)(it)
          exprEnv = ChildEnv.tableAggregateByKey(BindingEnv.empty[Type], 1, FastSeq(child))(snd)
          expr <- ir_value_expr(ctx, exprEnv)(it)
        } yield TableAggregateByKey(child, expr)
      case "TableKeyByAndAggregate" =>
        val nPartitions = opt(it, int32_literal)
        val bufferSize = int32_literal(it)
        def childEnv(i: Int, cs: IndexedSeq[BaseIR]): BindingEnv[Type] =
          ChildEnv.tableKeyByAndAggregate(BindingEnv.empty[Type], i, cs)(snd)
        for {
          child <- table_ir(ctx)(it)
          expr <- ir_value_expr(ctx, childEnv(1, FastSeq(child)))(it)
          newKey <- ir_value_expr(ctx, childEnv(2, FastSeq(child, expr)))(it)
        } yield TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize)
      case "TableRepartition" =>
        val n = int32_literal(it)
        val strategy = int32_literal(it)
        table_ir(ctx)(it).map(child => TableRepartition(child, n, strategy))
      case "TableHead" =>
        val n = int64_literal(it)
        table_ir(ctx)(it).map(child => TableHead(child, n))
      case "TableTail" =>
        val n = int64_literal(it)
        table_ir(ctx)(it).map(child => TableTail(child, n))
      case "TableJoin" =>
        val joinType = identifier(it)
        val joinKey = int32_literal(it)
        for {
          left <- table_ir(ctx)(it)
          right <- table_ir(ctx)(it)
        } yield TableJoin(left, right, joinType, joinKey)
      case "TableLeftJoinRightDistinct" =>
        val root = identifier(it)
        for {
          left <- table_ir(ctx)(it)
          right <- table_ir(ctx)(it)
        } yield TableLeftJoinRightDistinct(left, right, root)
      case "TableIntervalJoin" =>
        val root = identifier(it)
        val product = boolean_literal(it)
        for {
          left <- table_ir(ctx)(it)
          right <- table_ir(ctx)(it)
        } yield TableIntervalJoin(left, right, root, product)
      case "TableMultiWayZipJoin" =>
        val dataName = string_literal(it)
        val globalsName = string_literal(it)
        table_ir_children(ctx)(it).map { children =>
          TableMultiWayZipJoin(children, dataName, globalsName)
        }
      case "TableParallelize" =>
        val nPartitions = opt(it, int32_literal)
        ir_value_expr(ctx, BindingEnv.empty)(it).map(rowsAndGlobal =>
          TableParallelize(rowsAndGlobal, nPartitions)
        )
      case "TableMapRows" =>
        for {
          child <- table_ir(ctx)(it)
          newRowEnv = ChildEnv.tableMapRows(BindingEnv.empty[Type], 1, FastSeq(child))(snd)
          newRow <- ir_value_expr(ctx, newRowEnv)(it)
        } yield TableMapRows(child, newRow)
      case "TableMapGlobals" =>
        for {
          child <- table_ir(ctx)(it)
          newRowEnv = ChildEnv.tableMapGlobals(BindingEnv.empty[Type], 1, FastSeq(child))(snd)
          newRow <- ir_value_expr(ctx, newRowEnv)(it)
        } yield TableMapGlobals(child, newRow)
      case "TableRange" =>
        val n = int32_literal(it)
        val nPartitions = opt(it, int32_literal)
        done(TableRange(n, nPartitions.getOrElse(ctx.backend.defaultParallelism)))
      case "TableUnion" => table_ir_children(ctx)(it).map(TableUnion(_))
      case "TableOrderBy" =>
        val sortFields = sort_fields(it)
        table_ir(ctx)(it).map(child => TableOrderBy(child, sortFields))
      case "TableExplode" =>
        val path = string_literals(it)
        table_ir(ctx)(it).map(child => TableExplode(child, path))
      case "CastMatrixToTable" =>
        val entriesField = string_literal(it)
        val colsField = string_literal(it)
        matrix_ir(ctx)(it).map(child => CastMatrixToTable(child, entriesField, colsField))
      case "MatrixToTableApply" =>
        val config = string_literal(it)
        matrix_ir(ctx)(it).map { child =>
          MatrixToTableApply(child, RelationalFunctions.lookupMatrixToTable(ctx, config))
        }
      case "TableToTableApply" =>
        val config = string_literal(it)
        table_ir(ctx)(it).map { child =>
          TableToTableApply(child, RelationalFunctions.lookupTableToTable(ctx, config))
        }
      case "BlockMatrixToTableApply" =>
        val config = string_literal(it)
        for {
          bm <- blockmatrix_ir(ctx)(it)
          aux <- ir_value_expr(ctx, BindingEnv.empty)(it)
        } yield BlockMatrixToTableApply(
          bm,
          aux,
          RelationalFunctions.lookupBlockMatrixToTable(ctx, config),
        )
      case "BlockMatrixToTable" => blockmatrix_ir(ctx)(it).map(BlockMatrixToTable)
      case "TableRename" =>
        val rowK = string_literals(it)
        val rowV = string_literals(it)
        val globalK = string_literals(it)
        val globalV = string_literals(it)
        table_ir(ctx)(it).map { child =>
          TableRename(child, rowK.zip(rowV).toMap, globalK.zip(globalV).toMap)
        }

      case "TableGen" =>
        val cname = name(it)
        val gname = name(it)
        val partitioner =
          between(punctuation(_, "("), punctuation(_, ")"), partitioner_literal(ctx))(it)
        val errorId = int32_literal(it)
        for {
          contexts <- ir_value_expr(ctx, BindingEnv.empty)(it)
          globals <- ir_value_expr(ctx, BindingEnv.empty)(it)
          bodyEnv = ChildEnv.tableGen(
            BindingEnv.empty[Type],
            2,
            cname,
            gname,
            FastSeq(contexts, globals),
          )(snd)
          body <- ir_value_expr(ctx, bodyEnv)(it)
        } yield TableGen(contexts, globals, cname, gname, body, partitioner, errorId)

      case "TableFilterIntervals" =>
        val keyType = type_expr(ctx.references)(it)
        val intervals = string_literal(it)
        val keep = boolean_literal(it)
        table_ir(ctx)(it).map { child =>
          TableFilterIntervals(
            child,
            JSONAnnotationImpex.importAnnotation(
              JsonMethods.parse(intervals),
              TArray(TInterval(keyType)),
              padNulls = false,
            ).asInstanceOf[IndexedSeq[Interval]],
            keep,
          )
        }
      case "TableMapPartitions" =>
        val globalsName = name(it)
        val partitionStreamName = name(it)
        val requestedKey = int32_literal(it)
        val allowedOverlap = int32_literal(it)
        for {
          child <- table_ir(ctx)(it)
          bodyEnv =
            ChildEnv.tableMapPartitions(
              BindingEnv.empty[Type],
              1,
              globalsName,
              partitionStreamName,
              FastSeq(child),
            )(snd)
          body <- ir_value_expr(ctx, bodyEnv)(it)
        } yield TableMapPartitions(child, globalsName, partitionStreamName, body, requestedKey,
          allowedOverlap)
      case "JavaTable" =>
        val id = int32_literal(it)
        done(ctx.PersistedIrCache(id).deepCopy.asInstanceOf[TableIR])
    }
  }

  def matrix_ir_children(ctx: ExecuteContext)(it: TokenIterator): StackFrame[IndexedSeq[MatrixIR]] =
    repUntil(it, matrix_ir(ctx), T.Punc(")"))

  def matrix_ir(ctx: ExecuteContext)(it: TokenIterator): StackFrame[MatrixIR] = {
    punctuation(it, "(")
    for {
      ir <- call(matrix_ir_1(ctx)(it))
      _ = punctuation(it, ")")
    } yield ir
  }

  def matrix_ir_1(ctx: ExecuteContext)(it: TokenIterator): StackFrame[MatrixIR] = {
    identifier(it) match {
      case "MatrixFilterCols" =>
        for {
          child <- matrix_ir(ctx)(it)
          predEnv = ChildEnv.matrixFilterCols(BindingEnv.empty[Type], 1, FastSeq(child))(snd)
          pred <- ir_value_expr(ctx, predEnv)(it)
        } yield MatrixFilterCols(child, pred)
      case "MatrixFilterRows" =>
        for {
          child <- matrix_ir(ctx)(it)
          predEnv = ChildEnv.matrixFilterRows(BindingEnv.empty[Type], 1, FastSeq(child))(snd)
          pred <- ir_value_expr(ctx, predEnv)(it)
        } yield MatrixFilterRows(child, pred)
      case "MatrixFilterEntries" =>
        for {
          child <- matrix_ir(ctx)(it)
          predEnv = ChildEnv.matrixFilterEntries(BindingEnv.empty[Type], 1, FastSeq(child))(snd)
          pred <- ir_value_expr(ctx, predEnv)(it)
        } yield MatrixFilterEntries(child, pred)
      case "MatrixMapCols" =>
        val newKey = opt(it, string_literals)
        for {
          child <- matrix_ir(ctx)(it)
          newColEnv = ChildEnv.matrixMapCols(BindingEnv.empty[Type], 1, FastSeq(child))(snd)
          newCol <- ir_value_expr(ctx, newColEnv)(it)
        } yield MatrixMapCols(child, newCol, newKey)
      case "MatrixKeyRowsBy" =>
        val key = identifiers(it)
        val isSorted = boolean_literal(it)
        matrix_ir(ctx)(it).map(child => MatrixKeyRowsBy(child, key, isSorted))
      case "MatrixMapRows" =>
        for {
          child <- matrix_ir(ctx)(it)
          newRowEnv = ChildEnv.matrixMapRows(BindingEnv.empty[Type], 1, FastSeq(child))(snd)
          newRow <- ir_value_expr(ctx, newRowEnv)(it)
        } yield MatrixMapRows(child, newRow)
      case "MatrixMapEntries" =>
        for {
          child <- matrix_ir(ctx)(it)
          newEntryEnv = ChildEnv.matrixMapEntries(BindingEnv.empty[Type], 1, FastSeq(child))(snd)
          newEntry <- ir_value_expr(ctx, newEntryEnv)(it)
        } yield MatrixMapEntries(child, newEntry)
      case "MatrixUnionCols" =>
        val joinType = identifier(it)
        for {
          left <- matrix_ir(ctx)(it)
          right <- matrix_ir(ctx)(it)
        } yield MatrixUnionCols(left, right, joinType)
      case "MatrixMapGlobals" =>
        for {
          child <- matrix_ir(ctx)(it)
          newGlobalsEnv = ChildEnv.matrixMapGlobals(BindingEnv.empty[Type], 1, FastSeq(child))(snd)
          newGlobals <- ir_value_expr(ctx, newGlobalsEnv)(it)
        } yield MatrixMapGlobals(child, newGlobals)
      case "MatrixAggregateColsByKey" =>
        def childEnv(i: Int, cs: IndexedSeq[BaseIR]): BindingEnv[Type] =
          ChildEnv.matrixAggregateColsByKey(BindingEnv.empty[Type], i, cs)(snd)
        for {
          child <- matrix_ir(ctx)(it)
          entryExpr <- ir_value_expr(ctx, childEnv(1, FastSeq(child)))(it)
          colExpr <- ir_value_expr(ctx, childEnv(2, FastSeq(child, entryExpr)))(it)
        } yield MatrixAggregateColsByKey(child, entryExpr, colExpr)
      case "MatrixAggregateRowsByKey" =>
        def childEnv(i: Int, cs: IndexedSeq[BaseIR]): BindingEnv[Type] =
          ChildEnv.matrixAggregateRowsByKey(BindingEnv.empty[Type], i, cs)(snd)
        for {
          child <- matrix_ir(ctx)(it)
          entryExpr <- ir_value_expr(ctx, childEnv(1, FastSeq(child)))(it)
          rowExpr <- ir_value_expr(ctx, childEnv(2, FastSeq(child, entryExpr)))(it)
        } yield MatrixAggregateRowsByKey(child, entryExpr, rowExpr)
      case "MatrixRead" =>
        val requestedTypeRaw = it.head match {
          case T.Ident(v @ ("None" | "DropColUIDs" | "DropRowUIDs" | "DropRowColUIDs")) =>
            it.next(): Unit
            Left(v)
          case _ =>
            Right(matrix_type_expr(ctx.references)(it))
        }
        val dropCols = boolean_literal(it)
        val dropRows = boolean_literal(it)
        val readerStr = string_literal(it)
        val reader = MatrixReader.fromJson(ctx, JsonMethods.parse(readerStr).asInstanceOf[JObject])
        val fullType = reader.fullMatrixType
        val requestedType = requestedTypeRaw match {
          case Left("None") => fullType
          case Left("DropRowUIDs") => fullType.copy(
              rowType = fullType.rowType.deleteKey(reader.rowUIDFieldName)
            )
          case Left("DropColUIDs") => fullType.copy(
              colType = fullType.colType.deleteKey(reader.colUIDFieldName)
            )
          case Left("DropRowColUIDs") => fullType.copy(
              rowType = fullType.rowType.deleteKey(reader.rowUIDFieldName),
              colType = fullType.colType.deleteKey(reader.colUIDFieldName),
            )
          case Right(t) => t
        }
        done(MatrixRead(requestedType, dropCols, dropRows, reader))
      case "MatrixAnnotateRowsTable" =>
        val root = string_literal(it)
        val product = boolean_literal(it)
        for {
          child <- matrix_ir(ctx)(it)
          table <- table_ir(ctx)(it)
        } yield MatrixAnnotateRowsTable(child, table, root, product)
      case "MatrixAnnotateColsTable" =>
        val root = string_literal(it)
        for {
          child <- matrix_ir(ctx)(it)
          table <- table_ir(ctx)(it)
        } yield MatrixAnnotateColsTable(child, table, root)
      case "MatrixExplodeRows" =>
        val path = identifiers(it)
        matrix_ir(ctx)(it).map(child => MatrixExplodeRows(child, path))
      case "MatrixExplodeCols" =>
        val path = identifiers(it)
        matrix_ir(ctx)(it).map(child => MatrixExplodeCols(child, path))
      case "MatrixChooseCols" =>
        val oldIndices = int32_literals(it)
        matrix_ir(ctx)(it).map(child => MatrixChooseCols(child, oldIndices))
      case "MatrixCollectColsByKey" =>
        matrix_ir(ctx)(it).map(MatrixCollectColsByKey)
      case "MatrixRepartition" =>
        val n = int32_literal(it)
        val strategy = int32_literal(it)
        matrix_ir(ctx)(it).map(child => MatrixRepartition(child, n, strategy))
      case "MatrixUnionRows" => matrix_ir_children(ctx)(it).map(MatrixUnionRows(_))
      case "MatrixDistinctByRow" => matrix_ir(ctx)(it).map(MatrixDistinctByRow)
      case "MatrixRowsHead" =>
        val n = int64_literal(it)
        matrix_ir(ctx)(it).map(child => MatrixRowsHead(child, n))
      case "MatrixColsHead" =>
        val n = int32_literal(it)
        matrix_ir(ctx)(it).map(child => MatrixColsHead(child, n))
      case "MatrixRowsTail" =>
        val n = int64_literal(it)
        matrix_ir(ctx)(it).map(child => MatrixRowsTail(child, n))
      case "MatrixColsTail" =>
        val n = int32_literal(it)
        matrix_ir(ctx)(it).map(child => MatrixColsTail(child, n))
      case "CastTableToMatrix" =>
        val entriesField = identifier(it)
        val colsField = identifier(it)
        val colKey = identifiers(it)
        table_ir(ctx)(it).map(child => CastTableToMatrix(child, entriesField, colsField, colKey))
      case "MatrixToMatrixApply" =>
        val config = string_literal(it)
        matrix_ir(ctx)(it).map { child =>
          MatrixToMatrixApply(child, RelationalFunctions.lookupMatrixToMatrix(ctx, config))
        }
      case "MatrixRename" =>
        val globalK = string_literals(it)
        val globalV = string_literals(it)
        val colK = string_literals(it)
        val colV = string_literals(it)
        val rowK = string_literals(it)
        val rowV = string_literals(it)
        val entryK = string_literals(it)
        val entryV = string_literals(it)
        matrix_ir(ctx)(it).map { child =>
          MatrixRename(
            child,
            globalK.zip(globalV).toMap,
            colK.zip(colV).toMap,
            rowK.zip(rowV).toMap,
            entryK.zip(entryV).toMap,
          )
        }
      case "MatrixFilterIntervals" =>
        val keyType = type_expr(ctx.references)(it)
        val intervals = string_literal(it)
        val keep = boolean_literal(it)
        matrix_ir(ctx)(it).map { child =>
          MatrixFilterIntervals(
            child,
            JSONAnnotationImpex.importAnnotation(
              JsonMethods.parse(intervals),
              TArray(TInterval(keyType)),
              padNulls = false,
            ).asInstanceOf[IndexedSeq[Interval]],
            keep,
          )
        }
    }
  }

  def blockmatrix_sparsifier(ctx: ExecuteContext)(it: TokenIterator)
    : StackFrame[BlockMatrixSparsifier] = {
    punctuation(it, "(")
    val lower = LoweringPipeline.relationalLowerer
    identifier(it) match {
      case "PyRowIntervalSparsifier" =>
        val blocksOnly = boolean_literal(it)
        punctuation(it, ")")
        ir_value_expr(ctx, BindingEnv.empty)(it).map { ir_ =>
          val ir = NormalizeNames()(ctx, ir_).asInstanceOf[IR]
          val Row(starts: IndexedSeq[Long @unchecked], stops: IndexedSeq[Long @unchecked]) =
            CompileAndEvaluate[Row](ctx, ir, lower = lower)
          RowIntervalSparsifier(blocksOnly, starts, stops)
        }
      case "PyBandSparsifier" =>
        val blocksOnly = boolean_literal(it)
        punctuation(it, ")")
        ir_value_expr(ctx, BindingEnv.empty)(it).map { ir_ =>
          val ir = NormalizeNames()(ctx, ir_).asInstanceOf[IR]
          val Row(l: Long, u: Long) = CompileAndEvaluate[Row](ctx, ir, lower = lower)
          BandSparsifier(blocksOnly, l, u)
        }
      case "PyPerBlockSparsifier" =>
        punctuation(it, ")")
        ir_value_expr(ctx, BindingEnv.empty)(it).map { ir_ =>
          val ir = NormalizeNames()(ctx, ir_).asInstanceOf[IR]
          val indices = CompileAndEvaluate[IndexedSeq[Int]](ctx, ir, lower = lower)
          PerBlockSparsifier(indices)
        }
      case "PyRectangleSparsifier" =>
        punctuation(it, ")")
        ir_value_expr(ctx, BindingEnv.empty)(it).map { ir_ =>
          val ir = NormalizeNames()(ctx, ir_).asInstanceOf[IR]
          val rectangles = CompileAndEvaluate[IndexedSeq[Long]](ctx, ir, lower = lower)
          RectangleSparsifier(rectangles.grouped(4).toIndexedSeq)
        }
      case "RowIntervalSparsifier" =>
        val blocksOnly = boolean_literal(it)
        val starts = int64_literals(it)
        val stops = int64_literals(it)
        punctuation(it, ")")
        done(RowIntervalSparsifier(blocksOnly, starts, stops))
      case "BandSparsifier" =>
        val blocksOnly = boolean_literal(it)
        val l = int64_literal(it)
        val u = int64_literal(it)
        punctuation(it, ")")
        done(BandSparsifier(blocksOnly, l, u))
      case "RectangleSparsifier" =>
        val rectangles = int64_literals(it)
        punctuation(it, ")")
        done(RectangleSparsifier(rectangles.grouped(4).toIndexedSeq))
    }
  }

  def blockmatrix_ir(ctx: ExecuteContext)(it: TokenIterator): StackFrame[BlockMatrixIR] = {
    punctuation(it, "(")
    for {
      ir <- call(blockmatrix_ir1(ctx)(it))
      _ = punctuation(it, ")")
    } yield ir
  }

  def blockmatrix_ir1(ctx: ExecuteContext)(it: TokenIterator): StackFrame[BlockMatrixIR] = {
    identifier(it) match {
      case "BlockMatrixRead" =>
        val readerStr = string_literal(it)
        val reader = BlockMatrixReader.fromJValue(ctx, JsonMethods.parse(readerStr))
        done(BlockMatrixRead(reader))
      case "BlockMatrixMap" =>
        val n = name(it)
        val needs_dense = boolean_literal(it)
        for {
          child <- blockmatrix_ir(ctx)(it)
          fEnv = ChildEnv.blockMatrixMap(BindingEnv.empty[Type], 1, n)(snd)
          f <- ir_value_expr(ctx, fEnv)(it)
        } yield BlockMatrixMap(child, n, f, needs_dense)
      case "BlockMatrixMap2" =>
        val lName = name(it)
        val rName = name(it)
        val sparsityStrategy = SparsityStrategy.fromString(identifier(it))
        for {
          left <- blockmatrix_ir(ctx)(it)
          right <- blockmatrix_ir(ctx)(it)
          fEnv = ChildEnv.blockMatrixMap2(BindingEnv.empty[Type], 2, lName, rName)(snd)
          f <- ir_value_expr(ctx, fEnv)(it)
        } yield BlockMatrixMap2(left, right, lName, rName, f, sparsityStrategy)
      case "BlockMatrixDot" =>
        for {
          left <- blockmatrix_ir(ctx)(it)
          right <- blockmatrix_ir(ctx)(it)
        } yield BlockMatrixDot(left, right)
      case "BlockMatrixBroadcast" =>
        val inIndexExpr = int32_literals(it)
        val shape = int64_literals(it)
        val blockSize = int32_literal(it)
        blockmatrix_ir(ctx)(it).map { child =>
          BlockMatrixBroadcast(child, inIndexExpr, shape, blockSize)
        }
      case "BlockMatrixAgg" =>
        val outIndexExpr = int32_literals(it)
        blockmatrix_ir(ctx)(it).map(child => BlockMatrixAgg(child, outIndexExpr))
      case "BlockMatrixFilter" =>
        val indices = literals(literals(int64_literal))(it)
        blockmatrix_ir(ctx)(it).map(child => BlockMatrixFilter(child, indices))
      case "BlockMatrixDensify" =>
        blockmatrix_ir(ctx)(it).map(BlockMatrixDensify)
      case "BlockMatrixSparsify" =>
        for {
          sparsifier <- blockmatrix_sparsifier(ctx)(it)
          child <- blockmatrix_ir(ctx)(it)
        } yield BlockMatrixSparsify(child, sparsifier)
      case "BlockMatrixSlice" =>
        val slices = literals(literals(int64_literal))(it)
        blockmatrix_ir(ctx)(it).map(child => BlockMatrixSlice(child, slices))
      case "ValueToBlockMatrix" =>
        val shape = int64_literals(it)
        val blockSize = int32_literal(it)
        ir_value_expr(ctx, BindingEnv.empty)(it).map(child =>
          ValueToBlockMatrix(child, shape, blockSize)
        )
      case "BlockMatrixRandom" =>
        val staticUID = int64_literal(it)
        val gaussian = boolean_literal(it)
        val shape = int64_literals(it)
        val blockSize = int32_literal(it)
        done(BlockMatrixRandom(staticUID, gaussian, shape, blockSize))
    }
  }

  def parse[A](s: String, f: (TokenIterator) => A): A = {
    val it = new IRLexer(s).buffered
    val result = f(it)
    if (it.hasNext) error(it.head, s"Unexpected trailing ${it.head.kind} '${it.head.value}'.")
    result
  }

  def parse_value_ir(ctx: ExecuteContext, s: String, typeEnv: BindingEnv[Type] = BindingEnv.empty)
    : IR =
    TimedBlock.enter {
      val ir = parse(s, ir_value_expr(ctx, typeEnv)(_).run())
      TypeCheck(ctx, ir, typeEnv)
      ir
    }

  def parse_table_ir(ctx: ExecuteContext, s: String): TableIR =
    TimedBlock.enter {
      val ir = parse(s, table_ir(ctx)(_).run())
      TypeCheck(ctx, ir)
      ir
    }

  def parse_matrix_ir(ctx: ExecuteContext, s: String): MatrixIR =
    TimedBlock.enter {
      val ir = parse(s, matrix_ir(ctx)(_).run())
      TypeCheck(ctx, ir)
      ir
    }

  def parse_blockmatrix_ir(ctx: ExecuteContext, s: String): BlockMatrixIR =
    TimedBlock.enter {
      val ir = parse(s, blockmatrix_ir(ctx)(_).run())
      TypeCheck(ctx, ir)
      ir
    }

  def parseType(code: String, rgs: ReferenceGenomeEnv): Type = parse(code, type_expr(rgs))

  def parsePType(code: String, rgs: ReferenceGenomeEnv): PType = parse(code, ptype_expr(rgs))

  def parseStructType(code: String, rgs: ReferenceGenomeEnv): TStruct =
    tcoerce[TStruct](parse(code, type_expr(rgs)))

  def parseRVDType(code: String, rgs: ReferenceGenomeEnv): RVDType =
    parse(code, rvd_type_expr(rgs))

  def parseTableType(code: String, rgs: ReferenceGenomeEnv): TableType =
    parse(code, table_type_expr(rgs))

  def parseMatrixType(code: String, rgs: ReferenceGenomeEnv): MatrixType =
    parse(code, matrix_type_expr(rgs))

  def parseSortField(code: String): SortField = parse(code, sort_field)
}
