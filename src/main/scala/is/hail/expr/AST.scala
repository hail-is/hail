package is.hail.expr

import is.hail.expr.ir.IR
import is.hail.asm4s.{Code, _}
import is.hail.utils.EitherIsAMonad._
import is.hail.utils.{HailException, _}
import is.hail.variant.GenomeReference
import org.apache.spark.sql.{Row, RowFactory}
import org.json4s.jackson.JsonMethods

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.language.existentials
import scala.reflect.ClassTag
import scala.util.parsing.input.{Position, Positional}

case class RefBox(var v: Any)

case class EvalContext private(st: SymbolTable,
  a: ArrayBuffer[Any],
  aggregations: ArrayBuffer[(RefBox, CPS[Any], Aggregator)]) {
  st.foreach {
    case (name, (i, t: TAggregable)) =>
      require(t.symTab.exists { case (_, (j, t2)) => j == i && t2 == t.elementType },
        s"did not find binding for type ${ t.elementType } at index $i in agg symbol table for `$name'")
    case _ =>
  }

  def setAllFromRows(args1: Row, args2: Row) {
    var i = 0
    while (i < args1.length) {
      a(i) = args1(i)
      i += 1
    }
    var j = 0
    while (j < args2.length) {
      a(i) = args2(j)
      i += 1
      j += 1
    }
  }

  def setAllFromRow(args: Row) {
    var i = 0
    while (i < args.length) {
      a(i) = args(i)
      i += 1
    }
  }

  def setAll(arg1: Any) {
    a(0) = arg1
  }

  def setAll(arg1: Any, arg2: Any) {
    a(0) = arg1
    a(1) = arg2
  }

  def setAll(arg1: Any, arg2: Any, arg3: Any) {
    a(0) = arg1
    a(1) = arg2
    a(2) = arg3
  }

  def setAll(arg1: Any, arg2: Any, arg3: Any, arg4: Any) {
    a(0) = arg1
    a(1) = arg2
    a(2) = arg3
    a(3) = arg4
  }

  def setAll(arg1: Any, arg2: Any, arg3: Any, arg4: Any, arg5: Any) {
    a(0) = arg1
    a(1) = arg2
    a(2) = arg3
    a(3) = arg4
    a(4) = arg5
  }

  def setAll(arg1: Any, arg2: Any, arg3: Any, arg4: Any, arg5: Any, args: Any*) {
    a(0) = arg1
    a(1) = arg2
    a(2) = arg3
    a(3) = arg4
    a(4) = arg5
    var i = 0
    while (i < args.length) {
      a(5 + i) = args(i)
      i += 1
    }
  }

  def set(index: Int, arg: Any) {
    a(index) = arg
  }
}

object EvalContext {
  def apply(symTab0: SymbolTable): EvalContext = {
    val symTab = symTab0.map { case (name, (i, t)) =>
      (name, (i, t.deepOptional()))
    }

    def maxEntry(st: SymbolTable): Int = {
      val m = st.map {
        case (name, (i, t: TAggregable)) => i.max(maxEntry(t.symTab))
        case (name, (i, t)) => i
      }

      if (m.isEmpty)
        -1
      else
        m.max
    }

    val m = maxEntry(symTab) + 1
    val a = ArrayBuffer.fill[Any](m)(null)
    val af = new ArrayBuffer[(RefBox, CPS[Any], Aggregator)]()
    EvalContext(symTab, a, af)
  }

  def apply(args: (String, Type)*): EvalContext = {
    EvalContext(args.zipWithIndex
      .map { case ((name, t), i) => (name, (i, t.deepOptional())) }
      .toMap)
  }
}

sealed trait NumericConversion[T, U <: ToBoxed[T]] extends Serializable {
  def to(numeric: Any): T

  def to(c: Code[java.lang.Number]): CM[Code[U#T]]
}

object IntNumericConversion extends NumericConversion[Int, BoxedInt] {
  def to(numeric: Any): Int = numeric match {
    case i: Int => i
  }

  def to(c: Code[java.lang.Number]) =
    c.mapNull((x: Code[java.lang.Number]) => Code.boxInt(Code.intValue(x)))
}

object LongNumericConversion extends NumericConversion[Long, BoxedLong] {
  def to(numeric: Any): Long = numeric match {
    case i: Int => i
    case l: Long => l
  }

  def to(c: Code[java.lang.Number]) =
    c.mapNull((x: Code[java.lang.Number]) => Code.boxLong(Code.longValue(x)))
}

object FloatNumericConversion extends NumericConversion[Float, BoxedFloat] {
  def to(numeric: Any): Float = numeric match {
    case i: Int => i
    case l: Long => l
    case f: Float => f
  }

  def to(c: Code[java.lang.Number]) =
    c.mapNull((x: Code[java.lang.Number]) => Code.boxFloat(Code.floatValue(x)))
}

object DoubleNumericConversion extends NumericConversion[Double, BoxedDouble] {
  def to(numeric: Any): Double = numeric match {
    case i: Int => i
    case l: Long => l
    case f: Float => f
    case d: Double => d
  }

  def to(c: Code[java.lang.Number]) =
    c.mapNull((x: Code[java.lang.Number]) => Code.boxDouble(Code.doubleValue(x)))
}

object AST extends Positional {

  def evalComposeCode[T](subexpr: AST)(g: Code[T] => Code[AnyRef]): CM[Code[AnyRef]] =
    subexpr.compile().flatMap(_.mapNull((x: Code[AnyRef]) => g(x.asInstanceOf[Code[T]])))

  def evalComposeCodeM[T](subexpr: AST)(g: Code[T] => CM[Code[AnyRef]]): CM[Code[AnyRef]] = for (
    (stc, c) <- CM.memoize(subexpr.compile());
    gc <- g(c.asInstanceOf[Code[T]])
  ) yield
    Code(stc, c.ifNull(Code._null, gc))

  def evalComposeCodeM[T, U](a: AST, b: AST)(g: (Code[T], Code[U]) => CM[Code[AnyRef]]): CM[Code[AnyRef]] = for (
    (sta, ac) <- CM.memoize(a.compile());
    (stb, bc) <- CM.memoize(b.compile());
    gc <- g(ac.asInstanceOf[Code[T]], bc.asInstanceOf[Code[U]])
  ) yield
    Code(sta, ac.ifNull(Code._null,
      Code(stb, bc.ifNull(Code._null, gc))))

  def evalComposeCodeM[T, U, V](a: AST, b: AST, c: AST)(g: (Code[T], Code[U], Code[V]) => CM[Code[AnyRef]]): CM[Code[AnyRef]] = for (
    (sta, ac) <- CM.memoize(a.compile());
    (stb, bc) <- CM.memoize(b.compile());
    (stc, cc) <- CM.memoize(c.compile());
    gc <- g(ac.asInstanceOf[Code[T]], bc.asInstanceOf[Code[U]], cc.asInstanceOf[Code[V]]))
    yield
      Code(sta, ac.ifNull(Code._null,
        Code(stb, bc.ifNull(Code._null,
          Code(stc, cc.ifNull(Code._null, gc))))))

  def evalComposeCodeM[T, U, V, W](a: AST, b: AST, c: AST, d: AST)(g: (Code[T], Code[U], Code[V], Code[W]) => CM[Code[AnyRef]]): CM[Code[AnyRef]] = for (
    (sta, ac) <- CM.memoize(a.compile());
    (stb, bc) <- CM.memoize(b.compile());
    (stc, cc) <- CM.memoize(c.compile());
    (std, dc) <- CM.memoize(d.compile());
    gc <- g(ac.asInstanceOf[Code[T]], bc.asInstanceOf[Code[U]], cc.asInstanceOf[Code[V]], dc.asInstanceOf[Code[W]])
  ) yield
    Code(sta, ac.ifNull(Code._null,
      Code(stb, bc.ifNull(Code._null,
        Code(stc, cc.ifNull(Code._null,
          Code(std, dc.ifNull(Code._null, gc))))))))

  def evalComposeCodeM[T, U, V, W, X](a: AST, b: AST, c: AST, d: AST, e: AST)(g: (Code[T], Code[U], Code[V], Code[W], Code[X]) => CM[Code[AnyRef]]): CM[Code[AnyRef]] = for (
    (sta, ac) <- CM.memoize(a.compile());
    (stb, bc) <- CM.memoize(b.compile());
    (stc, cc) <- CM.memoize(c.compile());
    (std, dc) <- CM.memoize(d.compile());
    (ste, ec) <- CM.memoize(e.compile());
    gc <- g(ac.asInstanceOf[Code[T]], bc.asInstanceOf[Code[U]], cc.asInstanceOf[Code[V]], dc.asInstanceOf[Code[W]], ec.asInstanceOf[Code[X]])
  ) yield
    Code(sta, ac.ifNull(Code._null,
      Code(stb, bc.ifNull(Code._null,
        Code(stc, cc.ifNull(Code._null,
          Code(std, dc.ifNull(Code._null,
            Code(ste, ec.ifNull(Code._null, gc)
            )))))))))
}

case class Positioned[T](x: T) extends Positional

sealed abstract class AST(pos: Position, subexprs: Array[AST] = Array.empty) {
  var `type`: Type = _

  def getPos: Position = pos

  def this(posn: Position, subexpr1: AST) = this(posn, Array(subexpr1))

  def this(posn: Position, subexpr1: AST, subexpr2: AST) = this(posn, Array(subexpr1, subexpr2))

  def typecheckThis(ec: EvalContext): Type = typecheckThis()

  def typecheckThis(): Type = parseError(s"Found out-of-place expression of type ${ this.getClass.getSimpleName }")

  def typecheck(ec: EvalContext) {
    subexprs.foreach(_.typecheck(ec))
    `type` = typecheckThis(ec)
  }

  def parseError(msg: String): Nothing = ParserUtils.error(pos, msg)

  def errorIf[T <: AST](msg: String)(implicit c: ClassTag[T]) {
    if (c.runtimeClass.isInstance(this))
      parseError(msg)
    subexprs.foreach(_.errorIf[T](msg))
  }

  def compile(): CM[Code[AnyRef]]

  def compileAggregator(): CMCodeCPS[AnyRef]

  def run(ec: EvalContext): () => AnyRef = this.compile().run(ec)

  def runAggregator(ec: EvalContext): CPS[Any] = {
    val typedNames = ec.st.toSeq
      .sortBy { case (_, (i, _)) => i }
      .map { case (name, (_, typ)) => (name, typ) }
    val values = ec.a.asInstanceOf[mutable.ArrayBuffer[AnyRef]]

    val idx = ec.a.length
    val localA = ec.a
    localA += null

    val f = (this.compileAggregator() { (me: Code[AnyRef]) => for (
      vs <- CM.initialValueArray();
      k = Code.checkcast[AnyRef => Unit](vs.invoke[Int, AnyRef]("apply", idx))
    // This method returns `Void` which is only inhabited by `null`, we treat
    // these calls as non-stack-modifying functions so we must include a pop
    // to reset the stack.
    ) yield Code.toUnit(k.invoke[AnyRef, AnyRef]("apply", me))
    }).map(x => Code(x, Code._null[AnyRef])).runWithDelayedValues(typedNames, ec)

    { (k: (Any => Unit)) =>
      localA(idx) = k
      f(values)
    }
  }

  def toIR(agg: Option[String] = None): Option[IR]
}

case class Const(posn: Position, value: Any, t: Type) extends AST(posn) {
  def compileAggregator(): CMCodeCPS[AnyRef] = throw new UnsupportedOperationException

  def compile() = CM.ret(value match {
    case i: Int => Code.newInstance[java.lang.Integer, Int](i)
    case l: Long => Code.newInstance[java.lang.Long, Long](l)
    // case f: Float => (FloatInfo, f)
    case d: Double => Code.newInstance[java.lang.Double, Double](d)
    case s: String => (s: Code[String]).asInstanceOf[Code[AnyRef]]
    case z: Boolean => Code.newInstance[java.lang.Boolean, Boolean](z)
    // case c: Char => (CharInfo, c)
    case null => Code._null
    case _ => throw new RuntimeException(s"Unrecognized constant: $value")
  })

  override def typecheckThis(): Type = t

  def toIR(agg: Option[String] = None): Option[IR] = (t, value) match {
    case (t: TInt32, x: Int) => Some(ir.I32(x))
    case (t: TInt64, x: Long) => Some(ir.I64(x))
    case (t: TFloat32, x: Float) => Some(ir.F32(x))
    case (t: TFloat64, x: Double) => Some(ir.F64(x))
    case (t: TBoolean, x: Boolean) => Some(if (x) ir.True() else ir.False())
    case (t: TString, x: String) => None
    case (t, null) => Some(ir.NA(t))
    case _ => throw new RuntimeException(s"Unrecognized constant of type $t: $value")
  }
}

case class Select(posn: Position, lhs: AST, rhs: String) extends AST(posn, lhs) {
  override def typecheckThis(): Type = {
    (lhs.`type`, rhs) match {

      case (t: TStruct, _) =>
        t.selfField(rhs) match {
          case Some(f) => f.typ
          case None => parseError(
            s"""Struct has no field `$rhs'
               |  Available fields:
               |    ${ t.fields.map(x => s"${ prettyIdentifier(x.name) }: ${ x.typ }").mkString("\n    ") }""".stripMargin)
        }

      case (t, name) => FunctionRegistry.lookupFieldReturnType(t, Seq(), name)
        .valueOr {
          case FunctionRegistry.NotFound(name, typ, _) =>
            parseError(s"""`$t' has no field or method `$rhs'""".stripMargin)
          case otherwise => parseError(otherwise.message)
        }
    }
  }

  def compileAggregator(): CMCodeCPS[AnyRef] =
    FunctionRegistry.callAggregatorTransformation(lhs.`type`, Seq(), rhs)(lhs, Seq())

  def compile() = (lhs.`type`: @unchecked) match {
    case t: TStruct =>
      val Some(f) = t.selfField(rhs)
      val i = f.index
      AST.evalComposeCode[Row](lhs) { r: Code[Row] => Code.checkcast(r.invoke[Int, AnyRef]("get", i))(f.typ.scalaClassTag) }

    case t =>
      val localPos = posn

      FunctionRegistry.lookupField(t, Seq(), rhs)(lhs, Seq())
        .valueOr {
          case FunctionRegistry.NotFound(name, typ, _) =>
            ParserUtils.error(localPos,
              s"""`$t' has neither a field nor a method named `$name'
               |  Hint: sum, min, max, etc. have no parentheses when called on an Array:
               |    counts.sum""".stripMargin)
          case otherwise => fatal(otherwise.message)
        }
  }

  def toIR(agg: Option[String] = None): Option[IR] = for {
    s <- lhs.toIR(agg)
    t <- someIf(lhs.`type`.isInstanceOf[TStruct], lhs.`type`.asInstanceOf[TStruct])
    f <- t.selfField(rhs)
  } yield ir.GetField(s, rhs, f.typ)
}

case class ArrayConstructor(posn: Position, elements: Array[AST]) extends AST(posn, elements) {
  override def typecheckThis(ec: EvalContext): Type = {
    if (elements.isEmpty)
      parseError("invalid array constructor: no values")
    elements.foreach(_.typecheck(ec))
    val types: Set[Type] = elements.map(_.`type`)
      .map {
        case t: Type => t
        case bt => parseError(s"invalid array element found: `$bt'")
      }
      .toSet
    if (types.size == 1)
      TArray(types.head)
    else if (types.forall(_.isInstanceOf[TNumeric])) {
      TArray(TNumeric.promoteNumeric(types.map(_.asInstanceOf[TNumeric])))
    } else
      parseError(s"declared array elements must be the same type (or numeric)." +
        s"\n  Found: [${ elements.map(_.`type`).mkString(", ") }]")
  }

  def compileAggregator(): CMCodeCPS[AnyRef] = throw new UnsupportedOperationException

  def compile() = for (
    celements <- CM.sequence(elements.map(_.compile()));
    convertedArray <- CompilationHelp.arrayOfWithConversion(`type`.asInstanceOf[TArray].elementType, celements))
    yield
      CompilationHelp.arrayToWrappedArray(convertedArray)

  def toIR(agg: Option[String] = None): Option[IR] = for {
    irElements <- anyFailAllFail(elements.map(_.toIR(agg)))
  } yield ir.MakeArray(irElements.toArray, `type`.asInstanceOf[TArray])
}

case class StructConstructor(posn: Position, names: Array[String], elements: Array[AST]) extends AST(posn, elements) {
  override def typecheckThis(ec: EvalContext): Type = {
    elements.foreach(_.typecheck(ec))
    val types = elements.map(_.`type`)
      .map {
        case t: Type => t
        case bt => parseError(s"invalid struct element found: `$bt'")
      }
    TStruct((names, types, names.indices).zipped.map { case (id, t, i) => Field(id, t, i) })
  }

  def compileAggregator(): CMCodeCPS[AnyRef] = throw new UnsupportedOperationException

  private def arrayToAnnotation(c: Code[Array[AnyRef]]): Code[Row] =
    Code.invokeStatic[RowFactory, Array[AnyRef], Row]("create", c)

  def compile() = for (
    celements <- CM.sequence(elements.map(_.compile()))
  ) yield arrayToAnnotation(CompilationHelp.arrayOf(celements))

  def toIR(agg: Option[String] = None): Option[IR] = for {
    irElements <- anyFailAllFail[Array](elements.map(x => x.toIR(agg).map(ir => (x.`type`, ir))))
    fields = names.zip(irElements).map { case (x, (y, z)) => (x, z) }
  } yield ir.MakeStruct(fields)
}

case class GenomeReferenceDependentConstructor(posn: Position, fName: String, grName: String, args: Array[AST]) extends AST(posn, args) {
  val gr = GenomeReference.getReference(grName)
  val rTyp = fName match {
    case "Variant" => gr.variant
    case "Locus" => gr.locus
    case "Interval" => gr.interval
    case _ => throw new UnsupportedOperationException
  }

  override def typecheckThis(): Type = rTyp

  override def compileAggregator(): CMCodeCPS[AnyRef] = throw new UnsupportedOperationException

  override def compile(): CM[Code[AnyRef]] = FunctionRegistry.call(fName, args, args.map(_.`type`).toSeq, Some(rTyp))

  def toIR(agg: Option[String] = None): Option[IR] = None
}

case class Lambda(posn: Position, param: String, body: AST) extends AST(posn, body) {
  def typecheck(): Type = parseError("non-function context")

  def compileAggregator(): CMCodeCPS[AnyRef] = throw new UnsupportedOperationException

  def compile() = throw new UnsupportedOperationException

  def toIR(agg: Option[String] = None): Option[IR] = None
}

case class Apply(posn: Position, fn: String, args: Array[AST]) extends AST(posn, args) {
  override def typecheckThis(): Type = {
    (fn, args) match {
      case ("merge", rhs) =>
        val (t1, t2) = args.map(_.`type`) match {
          case Array(t1: TStruct, t2: TStruct) => (t1, t2)
          case other => parseError(
            s"""invalid arguments to `$fn'
               |  Expected $fn(Struct, Struct), found $fn(${ other.mkString(", ") })""".stripMargin)
        }

        val (t, _) = try {
          t1.merge(t2)
        } catch {
          case e: Throwable => parseError(
            s"""invalid arguments for method `$fn'
               |  $e""".stripMargin)
        }

        t

      case ("==" | "!=", Array(left, right)) =>
        val lt = left.`type`
        val rt = right.`type`
        if (!lt.canCompare(rt))
          parseError(s"Cannot compare arguments of type $lt and $rt")
        else
          TBoolean()
      case (_, _) => FunctionRegistry.lookupFunReturnType(fn, args.map(_.`type`).toSeq)
        .valueOr(x => parseError(x.message))
    }
  }

  override def typecheck(ec: EvalContext) {
    fn match {
      case "index" =>
        if (args.length != 2)
          parseError(
            s"""invalid arguments for method `$fn'
               |  Expected 2 arguments: $fn(Array[Struct], identifiers...)
               |  Found ${ args.length } arguments""".stripMargin)
        args.head.typecheck(ec)
        val t = args.head.`type` match {
          case TArray(t: TStruct, _) => t
          case error => parseError(
            s"""invalid arguments for method `$fn'
               |  Expected Array[Struct] as first argument, found `$error'""".stripMargin)
        }
        val key = args(1) match {
          case SymRef(_, id) => id
          case other =>
            parseError(
              s"""invalid arguments for method `$fn'
                 |  Expected struct field identifier as the second argument, but found a `${ other.getClass.getSimpleName }' expression
                 |  Usage: $fn(Array[Struct], key identifier)""".stripMargin)
        }

        t.getOption(key) match {
          case Some(keyType) =>
            val (newS, _) = t.delete(key)
            `type` = TDict(keyType, newS)
          case None => parseError(
            s"""invalid arguments for method `$fn'
               |  Struct did not contain the designated key `${ prettyIdentifier(key) }'""".stripMargin)
        }

      case "select" | "drop" =>
        if (args.length < 2)
          parseError(
            s"""too few arguments for method `$fn'
               |  Expected 2 or more arguments: $fn(Struct, identifiers...)
               |  Found ${ args.length } ${ plural(args.length, "argument") }""".stripMargin)
        val (head, tail) = (args.head, args.tail)
        head.typecheck(ec)
        val struct = head.`type` match {
          case t: TStruct => t
          case other => parseError(
            s"""method `$fn' expects a Struct argument in the first position
               |  Expected: $fn(Struct, ...)
               |  Found: $fn($other, ...)""".stripMargin)
        }
        val identifiers = tail.map {
          case SymRef(_, id) => id
          case other =>
            parseError(
              s"""invalid arguments for method `$fn'
                 |  Expected struct field identifiers after the first position, but found a `${ other.getClass.getSimpleName }' expression""".stripMargin)
        }
        val duplicates = identifiers.duplicates()
        if (duplicates.nonEmpty)
          parseError(
            s"""invalid arguments for method `$fn'
               |  Duplicate ${ plural(duplicates.size, "identifier") } found: [ ${ duplicates.map(prettyIdentifier).mkString(", ") } ]""".stripMargin)

        val (tNew, _) = try {
          struct.filter(identifiers.toSet, include = fn == "select")
        } catch {
          case e: Throwable => parseError(
            s"""invalid arguments for method `$fn'
               |  $e""".stripMargin)
        }

        `type` = tNew

      case "ungroup" =>
        if (args.length != 3)
          parseError(
            s"""invalid arguments for method `$fn'
               |  Usage: $fn(Struct, identifier, mangle)
               |  Found ${ args.length } ${ plural(args.length, "argument") }""".stripMargin)

        val Array(s, id, m) = args
        s.typecheck(ec)
        val struct = s.`type` match {
          case t: TStruct => t
          case other => parseError(
            s"""method `$fn' expects a Struct argument in the first position
               |  Expected: $fn(Struct, ...)
               |  Found: $fn($other, ...)""".stripMargin)
        }
        val identifier = id match {
          case SymRef(_, n) => n
          case other =>
            parseError(
              s"""invalid arguments for method `$fn'
                 |  Expected struct field identifier in the second position, but found a `${ other.getClass.getSimpleName }' expression""".stripMargin)
        }
        val mangle = m match {
              case Const(_, v, TBoolean(_)) => v.asInstanceOf[Boolean]
              case other =>
                parseError(
                  s"""invalid arguments for method `$fn'
                 |  Expected boolean argument in the third position, but found a `${ other.getClass.getSimpleName }' expression""".stripMargin)
        }

        val (tNew, _) = try {
          struct.ungroup(identifier, mangle)
        } catch {
          case e: Throwable => parseError(
            s"""invalid arguments for method `$fn'
               |  $e""".stripMargin)
        }

        `type` = tNew

      case "group" =>
        if (args.length < 3)
          parseError(
            s"""too few arguments for method `$fn'
               |  Expected 3 or more arguments: $fn(Struct, dest, identifiers...)
               |  Found ${ args.length } ${ plural(args.length, "argument") }""".stripMargin)
        val (head, tail) = (args.head, args.tail)
        head.typecheck(ec)
        val struct = head.`type` match {
          case t: TStruct => t
          case other => parseError(
            s"""method `$fn' expects a Struct argument in the first position
               |  Expected: $fn(Struct, ...)
               |  Found: $fn($other, ...)""".stripMargin)
        }
        val identifiers = tail.map {
          case SymRef(_, id) => id
          case other =>
            parseError(
              s"""invalid arguments for method `$fn'
                 |  Expected struct field identifiers after the first position, but found a `${ other.getClass.getSimpleName }' expression""".stripMargin)
        }

        val dest = identifiers.head
        val idsToGroup = identifiers.tail

        val duplicates = idsToGroup.duplicates()
        if (duplicates.nonEmpty)
          parseError(
            s"""invalid arguments for method `$fn'
               |  Duplicate ${ plural(duplicates.size, "identifier") } found: [ ${ duplicates.map(prettyIdentifier).mkString(", ") } ]""".stripMargin)

        val (tNew, _) = try {
          struct.group(dest, idsToGroup)
        } catch {
          case e: Throwable => parseError(
            s"""invalid arguments for method `$fn'
               |  $e""".stripMargin)
        }

        `type` = tNew

      case "uniroot" =>
        if (args.length != 3)
          parseError("wrong number of arguments to uniroot")

        args(0) match {
          case f@Lambda(_, param, body) =>
            body.typecheck(ec.copy(st = ec.st + ((param, (-1, TFloat64())))))
            f.`type` = TFunction(Array(TFloat64()), body.`type`)

          case _ =>
            fatal("first argument to uniroot must be lambda expression")
        }

        args(1).typecheck(ec)
        args(2).typecheck(ec)

        `type` = FunctionRegistry.lookupFunReturnType("uniroot", args.map(_.`type`))
          .valueOr(x => parseError(x.message))

      case _ => super.typecheck(ec)
    }
  }

  def compileAggregator(): CMCodeCPS[AnyRef] = throw new UnsupportedOperationException

  def compile() = ((fn, args): @unchecked) match {
    case ("merge", Array(struct1, struct2)) => for (
      f1 <- struct1.compile();
      f2 <- struct2.compile();
      (_, merger) = struct1.`type`.asInstanceOf[TStruct].merge(struct2.`type`.asInstanceOf[TStruct]);
      result <- CM.invokePrimitive2(merger)(f1, f2)
    ) yield result.asInstanceOf[Code[AnyRef]] // totally could be a problem

    case ("select" | "drop" | "group", Array(head, tail@_*)) =>
      val struct = head.`type`.asInstanceOf[TStruct]
      val identifiers = tail.map { ast =>
        (ast: @unchecked) match {
          case SymRef(_, id) => id
        }
      }

      val f = fn match {
        case "select" | "drop" => struct.filter(identifiers.toSet, include = fn == "select")._2
        case "group" => struct.group(identifiers.head, identifiers.tail.toArray)._2
      }

      AST.evalComposeCodeM[AnyRef](head)(CM.invokePrimitive1(f.asInstanceOf[(AnyRef) => AnyRef]))

    case ("ungroup", Array(s, id, m)) =>
      val struct = s.`type`.asInstanceOf[TStruct]
      val (identifier, mangle) = (id: @unchecked, m: @unchecked) match {
        case (SymRef(_, n), Const(_, v, _)) => (n, v.asInstanceOf[Boolean])
      }
      val f = struct.ungroup(identifier, mangle)._2
      AST.evalComposeCodeM[AnyRef](s)(CM.invokePrimitive1(f.asInstanceOf[(AnyRef) => AnyRef]))
      
    case ("index", Array(structArray, k)) =>
      val key = (k: @unchecked) match {
        case SymRef(_, id) => id
      }
      val t = structArray.`type`.asInstanceOf[TArray].elementType.asInstanceOf[TStruct]
      val querier = t.query(key)
      val (_, deleter) = t.delete(key)

      AST.evalComposeCodeM[AnyRef](structArray)(CM.invokePrimitive1 { is =>
        is.asInstanceOf[IndexedSeq[AnyRef]]
          .filter(_ != null)
          .map { r => (querier(r), deleter(r)) }
          .filter { case (k, v) => k != null }
          .toMap
      })

    case (_, _) =>
      FunctionRegistry.call(fn, args, args.map(_.`type`).toSeq)
  }

  private val tryPrimOpConversion: IndexedSeq[IR] => Option[IR] = flatLift {
    case IndexedSeq(x) => for {
      op <- ir.UnaryOp.fromString.lift(fn)
      t <- ir.UnaryOp.returnTypeOption(op, x.typ)
    } yield ir.ApplyUnaryPrimOp(op, x, t)
    case IndexedSeq(x, y) => for {
      op <- ir.BinaryOp.fromString.lift(fn)
      t <- ir.BinaryOp.returnTypeOption(op, x.typ, y.typ)
    } yield ir.ApplyBinaryPrimOp(op, x, y, t)
  }


  def toIR(agg: Option[String] = None): Option[IR] = for {
    irArgs <- anyFailAllFail(args.map(_.toIR(agg)))
    ir <- tryPrimOpConversion(irArgs)
  } yield ir
}

case class ApplyMethod(posn: Position, lhs: AST, method: String, args: Array[AST]) extends AST(posn, lhs +: args) {
  def getSymRefId(ast: AST): String = {
    ast match {
      case SymRef(_, id) => id
      case _ => ???
    }
  }

  override def typecheck(ec: EvalContext) {
    lhs.typecheck(ec)
    (lhs.`type`, method, args) match {

      case (it: TAggregable, _, Array(Lambda(_, param, body), rest@_*)) =>
        rest.foreach(_.typecheck(ec.copy(st = emptySymTab)))
        body.typecheck(ec.copy(st = it.symTab + ((param, (-1, it.elementType)))))
        val funType = TFunction(Array(it.elementType), body.`type`)
        `type` = FunctionRegistry.lookupMethodReturnType(it, funType +: rest.map(_.`type`), method)
          .valueOr(x => parseError(x.message))

      // no lambda
      case (it: TAggregable, _, _) =>
        args.foreach(_.typecheck(ec.copy(st = emptySymTab)))
        `type` = FunctionRegistry.lookupMethodReturnType(it, args.map(_.`type`), method)
          .valueOr(x => parseError(x.message))

      // FIXME: This isn't an extensible solution. We need something general that works for mapKeys, etc
      case (td: TDict, "mapValues", Array(Lambda(_, param, body), rest@_*)) =>
        rest.foreach(_.typecheck(ec))
        body.typecheck(ec.copy(st = ec.st + ((param, (-1, td.valueType)))))
        val funType = TFunction(Array(td.valueType), body.`type`)
        `type` = FunctionRegistry.lookupMethodReturnType(td, funType +: rest.map(_.`type`), method)
          .valueOr(x => parseError(x.message))

      // not aggregable: TIterable
      case (it: TContainer, _, Array(Lambda(_, param, body), rest@_*)) =>
        rest.foreach(_.typecheck(ec))
        body.typecheck(ec.copy(st = ec.st + ((param, (-1, it.elementType)))))
        val funType = TFunction(Array(it.elementType), body.`type`)
        `type` = FunctionRegistry.lookupMethodReturnType(it, funType +: rest.map(_.`type`), method)
          .valueOr(x => parseError(x.message))

      case _ =>
        super.typecheck(ec)
    }
  }

  override def typecheckThis(): Type = {
    val rhsTypes = args.map(_.`type`)
    (lhs.`type`, method, rhsTypes) match {
      case (t, _, _) =>
        FunctionRegistry.lookupMethodReturnType(t, args.map(_.`type`).toSeq, method)
          .valueOr(x => parseError(x.message))
    }
  }

  def compileAggregator(): CMCodeCPS[AnyRef] = {
    val t = lhs.`type`.asInstanceOf[TAggregable]
    args match {
      case (Array(Lambda(_, param, body), rest@_*)) =>
        val funType = TFunction(Array(t.elementType), body.`type`)

        FunctionRegistry.callAggregatorTransformation(t, funType +: rest.map(_.`type`), method)(lhs, args)
      case _ =>
        FunctionRegistry.callAggregatorTransformation(t, args.map(_.`type`).toSeq, method)(lhs, args)
    }
  }

  def compile() = ((lhs.`type`, method, args): @unchecked) match {
    case (td: TDict, "mapValues", Array(Lambda(_, param, body), rest@_*)) =>
      val funType = TFunction(Array(td.valueType), body.`type`)
      FunctionRegistry.call(method, lhs +: args, td +: funType +: rest.map(_.`type`))
    case (it: TContainer, _, Array(Lambda(_, param, body), rest@_*)) =>
      val funType = TFunction(Array(it.elementType), body.`type`)
      FunctionRegistry.call(method, lhs +: args, it +: funType +: rest.map(_.`type`))
    case (t, _, _) => FunctionRegistry.call(method, lhs +: args, t +: args.map(_.`type`))
  }

  def toIR(agg: Option[String] = None): Option[IR] = (method, args: IndexedSeq[AST]) match {
    case ("[]", IndexedSeq(rhs)) if lhs.`type`.isInstanceOf[TArray] =>
      for {
        a <- lhs.toIR(agg)
        i <- rhs.toIR(agg)
      } yield ir.ArrayRef(a, i, `type`)
    case ("sum", IndexedSeq()) =>
      lhs.toIR(agg).flatMap { a2 => Some(ir.AggSum(a2)) }
    case (m, IndexedSeq(Lambda(_, name, body))) =>
      for {
        a <- lhs.toIR(agg)
        b <- body.toIR(agg)
        result <- optMatch(m) {
          case "map" => ir.AggMap(a, name, b)
          case "filter" => ir.AggFilter(a, name, b)
          case "flatMap" => ir.AggFlatMap(a, name, b)
        }
      } yield result
    case _ => None
  }
}

case class Let(posn: Position, bindings: Array[(String, AST)], body: AST) extends AST(posn, bindings.map(_._2) :+ body) {

  override def typecheck(ec: EvalContext) {
    var symTab2 = ec.st
    for ((id, v) <- bindings) {
      v.typecheck(ec.copy(st = symTab2))
      symTab2 = symTab2 + (id -> (-1, v.`type`))
    }
    body.typecheck(ec.copy(st = symTab2))

    `type` = body.`type`
  }

  def compileAggregator(): CMCodeCPS[AnyRef] = throw new UnsupportedOperationException

  def compile() = CM.bindRepIn(bindings.map { case (name, expr) => (name, expr.`type`, expr.compile()) })(body.compile())

  def toIR(agg: Option[String] = None): Option[IR] = for {
    irBindings <- anyFailAllFail(bindings.map { case (x, y) => y.toIR(agg).map(irY => (x, irY)) })
    irBody <- body.toIR(agg)
  } yield irBindings.foldRight(irBody) { case ((name, v), x) => ir.Let(name, v, x, x.typ) }
}

case class SymRef(posn: Position, symbol: String) extends AST(posn) {
  override def typecheckThis(ec: EvalContext): Type = {
    ec.st.get(symbol) match {
      case Some((_, t)) => t
      case None =>
        val symbols = ec.st.toArray.sortBy(_._2._1).map { case (id, (_, t)) => s"${ prettyIdentifier(id) }: $t" }
        parseError(
          s"""symbol `$symbol' not found
             |  Available symbols:
             |    ${ symbols.mkString("\n    ") }""".stripMargin)
    }
  }

  def compileAggregator(): CMCodeCPS[AnyRef] = {
    val x: CM[Code[AnyRef]] = CM.lookup(symbol)


    ((k: Code[AnyRef] => CM[Code[Unit]]) => x.flatMap(k))
  }

  def compile() = CM.lookup(symbol)

  def toIR(agg: Option[String] = None): Option[IR] = agg match {
    case Some(x) if x == symbol => Some(ir.AggIn())
    case _ => Some(ir.Ref(symbol, `type`))
  }
}

case class If(pos: Position, cond: AST, thenTree: AST, elseTree: AST)
  extends AST(pos, Array(cond, thenTree, elseTree)) {
  override def typecheckThis(ec: EvalContext): Type = {
    if (!cond.`type`.isInstanceOf[TBoolean]) {
      parseError(s"an `if` expression's condition must have type Boolean")
    }
    (thenTree.`type`, elseTree.`type`) match {
      case (thenType, elseType) if thenType == elseType => thenType
      case (thenType: TNumeric, elseType: TNumeric) => TNumeric.promoteNumeric(Set(thenType, elseType))
      case _ =>
        parseError(s"expected same-type `then' and `else' clause, got `${ thenTree.`type` }' and `${ elseTree.`type` }'")
    }
  }

  def compileAggregator(): CMCodeCPS[AnyRef] = throw new UnsupportedOperationException

  def compile() = {
    val coerce: Code[AnyRef] => CM[Code[AnyRef]] = `type` match {
      case t: TNumeric => (x: Code[AnyRef]) => t.conv.to(x.asInstanceOf[Code[java.lang.Number]])
      case _ => CM.ret _
    }

    for (
      tc <- thenTree.compile();
      ec <- elseTree.compile();
      cc <- cond.compile();
      result <- cc.asInstanceOf[Code[java.lang.Boolean]].mapNullM((cc: Code[java.lang.Boolean]) =>
        coerce(cc.invoke[Boolean]("booleanValue").mux(tc, ec)))
    ) yield result
  }

  def toIR(agg: Option[String] = None): Option[IR] = for {
    condition <- cond.toIR(agg)
    consequent <- thenTree.toIR(agg)
    alternate <- elseTree.toIR(agg)
  } yield ir.If(condition, consequent, alternate, `type`)
}
