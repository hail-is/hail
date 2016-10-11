package org.broadinstitute.hail.expr

import org.apache.spark.sql.Row
import org.apache.spark.util.StatCounter
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.stats._
import org.broadinstitute.hail.utils.{FatalException, Interval}
import org.broadinstitute.hail.variant.{AltAllele, Genotype, Locus, Variant}
import org.json4s.jackson.JsonMethods

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.parsing.input.{Position, Positional}
import scala.language.existentials
import scala.reflect.ClassTag

case class EvalContext(st: SymbolTable, a: ArrayBuffer[Any], aggregationFunctions: ArrayBuffer[Aggregator]) {

  def setAll(args: Any*) {
    args.zipWithIndex.foreach { case (arg, i) => a(i) = arg }
  }

  def set(index: Int, arg: Any) {
    a(index) = arg
  }
}

object EvalContext {
  def apply(symTab: SymbolTable): EvalContext = {
    val a = new ArrayBuffer[Any]()
    val af = new ArrayBuffer[Aggregator]()
    for ((i, t) <- symTab.values) {
      if (i >= 0)
        a += null
    }

    EvalContext(symTab, a, af)
  }

  def apply(args: (String, Type)*): EvalContext = {
    val st = args.zipWithIndex
      .map { case ((name, t), i) => (name, (i, t)) }
      .toMap
    EvalContext(st)
  }
}

trait NumericConversion[T] extends Serializable {
  def to(numeric: Any): T
}

object IntNumericConversion extends NumericConversion[Int] {
  def to(numeric: Any): Int = numeric match {
    case i: Int => i
  }
}

object LongNumericConversion extends NumericConversion[Long] {
  def to(numeric: Any): Long = numeric match {
    case i: Int => i
    case l: Long => l
  }
}

object FloatNumericConversion extends NumericConversion[Float] {
  def to(numeric: Any): Float = numeric match {
    case i: Int => i
    case l: Long => l
    case f: Float => f
  }
}

object DoubleNumericConversion extends NumericConversion[Double] {
  def to(numeric: Any): Double = numeric match {
    case i: Int => i
    case l: Long => l
    case f: Float => f
    case d: Double => d
  }
}

object AST extends Positional {
  def promoteNumeric(t: TNumeric): BaseType = t

  def promoteNumeric(lhs: TNumeric, rhs: TNumeric): TNumeric =
    if (lhs == TDouble || rhs == TDouble)
      TDouble
    else if (lhs == TFloat || rhs == TFloat)
      TFloat
    else if (lhs == TLong || rhs == TLong)
      TLong
    else
      TInt

  def evalFlatCompose[T](ec: EvalContext, subexpr: AST)
    (g: (T) => Option[Any]): () => Any = {
    val f = subexpr.eval(ec)
    () => {
      val x = f()
      if (x != null)
        g(x.asInstanceOf[T]).orNull
      else
        null
    }
  }

  def evalCompose[T](ec: EvalContext, subexpr: AST)
    (g: (T) => Any): () => Any = {
    val f = subexpr.eval(ec)
    () => {
      val x = f()
      if (x != null)
        g(x.asInstanceOf[T])
      else
        null
    }
  }

  def evalCompose[T1, T2](ec: EvalContext, subexpr1: AST, subexpr2: AST)
    (g: (T1, T2) => Any): () => Any = {
    val f1 = subexpr1.eval(ec)
    val f2 = subexpr2.eval(ec)
    () => {
      val x = f1()
      if (x != null) {
        val y = f2()
        if (y != null)
          g(x.asInstanceOf[T1], y.asInstanceOf[T2])
        else
          null
      } else
        null
    }
  }

  def evalCompose[T1, T2, T3](ec: EvalContext, subexpr1: AST, subexpr2: AST, subexpr3: AST)
    (g: (T1, T2, T3) => Any): () => Any = {
    val f1 = subexpr1.eval(ec)
    val f2 = subexpr2.eval(ec)
    val f3 = subexpr3.eval(ec)
    () => {
      val x = f1()
      if (x != null) {
        val y = f2()
        if (y != null) {
          val z = f3()
          if (z != null)
            g(x.asInstanceOf[T1], y.asInstanceOf[T2], z.asInstanceOf[T3])
          else
            null
        } else
          null
      } else
        null
    }
  }

  def evalCompose[T1, T2, T3, T4](ec: EvalContext, subexpr1: AST, subexpr2: AST, subexpr3: AST, subexpr4: AST)
    (g: (T1, T2, T3, T4) => Any): () => Any = {
    val f1 = subexpr1.eval(ec)
    val f2 = subexpr2.eval(ec)
    val f3 = subexpr3.eval(ec)
    val f4 = subexpr4.eval(ec)
    () => {
      val w = f1()
      if (w != null) {
        val x = f2()
        if (x != null) {
          val y = f3()
          if (y != null) {
            val z = f4()
            if (z != null)
              g(w.asInstanceOf[T1], x.asInstanceOf[T2], y.asInstanceOf[T3], z.asInstanceOf[T4])
            else
              null
          } else
            null
        } else
          null
      } else
        null
    }
  }

  def evalNumeric[T](g: (T) => Any)(implicit conv: NumericConversion[T]): (Any) => Any = {
    (a: Any) => {
      if (a == null)
        null
      else g(conv.to(a))
    }
  }

  def evalComposeNumeric[T](ec: EvalContext, subexpr: AST)
    (g: (T) => Any)
    (implicit convT: NumericConversion[T]): () => Any = evalComposeThunkNumeric[T](g, subexpr.eval(ec))

  def evalComposeThunkNumeric[T](g: (T) => Any, f: () => Any)(implicit conv: NumericConversion[T]): () => Any = {
    () => evalNumeric[T](g).apply(f())
  }

  def evalNumeric[T1, T2](g: (T1, T2) => Any)(implicit conv1: NumericConversion[T1], conv2: NumericConversion[T2]): (Any, Any) => Any = {
    (a1: Any, a2: Any) => {
      if (a1 == null || a2 == null)
        null
      else g(conv1.to(a1), conv2.to(a2))
    }
  }

  def evalComposeThunkNumeric[T1, T2](g: (T1, T2) => Any, f1: () => Any, f2: () => Any)(implicit conv1: NumericConversion[T1], conv2: NumericConversion[T2]): () => Any = {
    () => evalNumeric[T1, T2](g).apply(f1(), f2())
  }

  def evalComposeNumeric[T1, T2](ec: EvalContext, subexpr1: AST, subexpr2: AST)
    (g: (T1, T2) => Any)
    (implicit convT1: NumericConversion[T1], convT2: NumericConversion[T2]): () => Any = evalComposeThunkNumeric[T1, T2](g, subexpr1.eval(ec), subexpr2.eval(ec))
}

case class Positioned[T](x: T) extends Positional

sealed abstract class AST(pos: Position, subexprs: Array[AST] = Array.empty) {
  var `type`: BaseType = null

  def this(posn: Position, subexpr1: AST) = this(posn, Array(subexpr1))

  def this(posn: Position, subexpr1: AST, subexpr2: AST) = this(posn, Array(subexpr1, subexpr2))

  def eval(ec: EvalContext): () => Any

  def typecheckThis(ec: EvalContext): BaseType = typecheckThis()

  def typecheckThis(): BaseType = throw new UnsupportedOperationException

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
}

case class Const(posn: Position, value: Any, t: BaseType) extends AST(posn) {
  def eval(c: EvalContext): () => Any = {
    val v = value
    () => v
  }

  override def typecheckThis(): BaseType = t
}

case class Select(posn: Position, lhs: AST, rhs: String) extends AST(posn, lhs) {
  override def typecheckThis(): BaseType = {
    (lhs.`type`, rhs) match {
      case (TSample, "id") => TString

      case (t: TStruct, _) =>
        t.selfField(rhs) match {
          case Some(f) => f.`type`
          case None => parseError(
            s"""`$t' has no field `$rhs'
                |  Available fields: [ ${ t.fields.map(x => prettyIdentifier(x.name)).mkString("\n  ") } ]""".stripMargin)
        }

      case (t: TArray, "length") => TInt
      case (t: TIterable, "size") => TInt
      case (t: TIterable, "isEmpty") => TBoolean
      case (t: TIterable, "toSet") => TSet(t.elementType)
      case (t: TIterable, "toArray") => TArray(t.elementType)
      case (t: TDict, "size") => TInt
      case (t: TDict, "isEmpty") => TBoolean
      case (TArray(elementType), "head") => elementType
      case (t@TArray(elementType), "tail") => t

      case (t, name) => FunctionRegistry.lookupFieldType(t, name)
        .getOrElse(parseError(
          s"""`$t' has no field `$rhs'
              |  Hint: Don't forget empty-parentheses in a method call, e.g.
              |    gs.filter(g => g.isCalledHomVar).collect()""".stripMargin))
    }
  }

  def eval(ec: EvalContext): () => Any = ((lhs.`type`, rhs): @unchecked) match {
    case (TSample, "id") => lhs.eval(ec)

    case (t: TStruct, _) =>
      val Some(f) = t.selfField(rhs)
      val i = f.index
      AST.evalCompose[Row](ec, lhs)(_.get(i))

    case (t: TArray, "length") => AST.evalCompose[Iterable[_]](ec, lhs)(_.size)
    case (t: TIterable, "size") => AST.evalCompose[Iterable[_]](ec, lhs)(_.size)
    case (t: TIterable, "isEmpty") => AST.evalCompose[Iterable[_]](ec, lhs)(_.isEmpty)
    case (t: TIterable, "toSet") => AST.evalCompose[Iterable[_]](ec, lhs)(_.toSet)
    case (t: TIterable, "toArray") => AST.evalCompose[Iterable[_]](ec, lhs)(_.toSeq)

    case (t: TDict, "size") => AST.evalCompose[Map[_, _]](ec, lhs)(_.size)
    case (t: TDict, "isEmpty") => AST.evalCompose[Map[_, _]](ec, lhs)(_.isEmpty)

    case (TArray(elementType), "head") =>
      AST.evalCompose[IndexedSeq[_]](ec, lhs)(_.head)
    case (t@TArray(elementType), "tail") =>
      AST.evalCompose[IndexedSeq[_]](ec, lhs)(_.tail)

    case (t, name) => FunctionRegistry.lookupField(ec)(t, name)(lhs)
      .getOrElse(fatal(s"""`$t' has no field `$name'
                          |  Hint: sum, min, max, etc. have no parentheses when called on an Array:
                          |    counts.sum""".stripMargin))
  }
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
    }
    else
      parseError(s"declared array elements must be the same type (or numeric)." +
        s"\n  Found: [${ elements.map(_.`type`).mkString(", ") }]")
  }

  def eval(ec: EvalContext): () => Any = {
    val f = elements.map(_.eval(ec))
    `type`.asInstanceOf[TArray].elementType match {
      case t: TNumeric => () => f.map(v => Option(v()).map(t.conv.to(_)).orNull): IndexedSeq[Any]
      case _ => () => f.map(_ ()): IndexedSeq[Any]
    }
  }
}

case class StructConstructor(posn: Position, names: Array[String], elements: Array[AST]) extends AST(posn, elements) {
  override def typecheckThis(ec: EvalContext): Type = {
    if (elements.isEmpty)
      parseError("Hail does not currently support declaring empty structs.")
    elements.foreach(_.typecheck(ec))
    val types = elements.map(_.`type`)
      .map {
        case t: Type => t
        case bt => parseError(s"invalid array element found: `$bt'")
      }
    TStruct((names, types, names.indices).zipped.map { case (id, t, i) => Field(id, t, i) })
  }

  def eval(ec: EvalContext): () => Any = {
    val f = elements.map(_.eval(ec))
    () => Annotation.fromSeq(f.map(_ ()))
  }
}

case class Lambda(posn: Position, param: String, body: AST) extends AST(posn, body) {
  def typecheck(): BaseType = parseError("non-function context")

  def eval(ec: EvalContext): () => Any = throw new UnsupportedOperationException
}

case class Apply(posn: Position, fn: String, args: Array[AST]) extends AST(posn, args) {
  override def typecheckThis(): BaseType = {
    (fn, args) match {
      case ("isMissing", Array(a)) =>
        if (!a.`type`.isInstanceOf[Type])
          parseError(s"Got invalid argument `${ a.`type` } to function `$fn'")
        TBoolean

      case ("isDefined", Array(a)) =>
        if (!a.`type`.isInstanceOf[Type])
          parseError(s"Got invalid argument `${ a.`type` } to function `$fn'")
        TBoolean

      case ("str", Array(a)) =>
        if (!a.`type`.isInstanceOf[Type])
          parseError(s"Got invalid argument `${ a.`type` } to function `$fn'")
        TString

      case ("json", Array(a)) =>
        if (!a.`type`.isInstanceOf[Type])
          parseError(s"Got invalid argument `${ a.`type` } to function `$fn'")
        TString

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
          case f: FatalException => parseError(
            s"""invalid arguments for method `$fn'
                |  ${ f.getMessage }""".stripMargin)
          case e: Throwable => parseError(
            s"""invalid arguments for method `$fn'
                |  ${ e.getClass.getName }: ${ e.getMessage }""".stripMargin)
        }

        t

      case ("isDefined" | "isMissing" | "str" | "json", _) => parseError(s"`$fn' takes one argument")

      case ("pow", _) => TDouble
        args.map(_.`type`) match {
          case Array(a: TNumeric, b: TNumeric) => TDouble
          case other =>
            parseError(
              s"""invalid arguments in call to $fn: ${ other.mkString(", ") }.
                  |  Expected $fn(Double)""".stripMargin)
        }

      case ("log", Array(a, b)) if a.`type`.isInstanceOf[TNumeric] && b.`type`.isInstanceOf[TNumeric] => TDouble

      case (_, _) => FunctionRegistry.lookupFunReturnType(fn, args.map(_.`type`).toSeq)
        .getOrElse(parseError(s"No function found with name `$fn' and argument ${ plural(args.size, "type") } `${ args.map(_.`type`).mkString(", ") }'"))
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
          case TArray(t: TStruct) => t
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
          case Some(TString) =>
            val (newS, _) = t.delete(key)
            `type` = TDict(newS)
          case Some(other) => parseError(
            s"""invalid arguments for method `$fn'
                |  Expected key to be of type String, but field ${ prettyIdentifier(key) } had type `$other'""".stripMargin)
          case None => parseError(
            s"""invalid arguments for method `$fn'
                |  Struct did not contain the designated key `${ prettyIdentifier(key) }'""".stripMargin)
        }

      case "select" | "drop" =>
        val usage = s"""Usage: `$fn'(Struct, identifiers...)"""
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
          case f: FatalException => parseError(
            s"""invalid arguments for method `$fn'
                |  ${ f.getMessage }""".stripMargin)
          case e: Throwable => parseError(
            s"""invalid arguments for method `$fn'
                |  ${ e.getClass.getName }: ${ e.getMessage }""".stripMargin)
        }

        `type` = tNew

      case _ => super.typecheck(ec)
    }
  }

  def eval(ec: EvalContext): () => Any = ((fn, args): @unchecked) match {
    case ("isMissing", Array(a)) =>
      val f = a.eval(ec)
      () => f() == null

    case ("isDefined", Array(a)) =>
      val f = a.eval(ec)
      () => f() != null

    case ("str", Array(a)) =>
      val t = a.`type`.asInstanceOf[Type]
      val f = a.eval(ec)
      () => t.str(f())

    case ("json", Array(a)) =>
      val t = a.`type`.asInstanceOf[Type]
      val f = a.eval(ec)
      () => JsonMethods.compact(t.toJSON(f()))

    case ("merge", Array(struct1, struct2)) =>
      val (_, merger) = struct1.`type`.asInstanceOf[TStruct].merge(struct2.`type`.asInstanceOf[TStruct])
      val f1 = struct1.eval(ec)
      val f2 = struct2.eval(ec)
      () => {
        merger(f1(), f2())
      }

    case ("select" | "drop", rhs) =>
      val (head, tail) = (rhs.head, rhs.tail)
      val struct = head.`type`.asInstanceOf[TStruct]
      val identifiers = tail.map { ast =>
        (ast: @unchecked) match {
          case SymRef(_, id) => id
        }
      }

      val (_, filterer) = struct.filter(identifiers.toSet, include = fn == "select")


      AST.evalCompose[Annotation](ec, head) { s =>
        filterer(s)
      }

    case ("index", Array(structArray, k)) =>
      val key = (k: @unchecked) match {
        case SymRef(_, id) => id
      }
      val t = structArray.`type`.asInstanceOf[TArray].elementType.asInstanceOf[TStruct]
      val querier = t.query(key)
      val (_, deleter) = t.delete(key)

      AST.evalCompose[IndexedSeq[_]](ec, structArray) { is =>
        is.filter(_ != null)
          .map(_.asInstanceOf[Row])
          .flatMap(r => querier(r).map(x => (x, deleter(r))))
          .toMap
      }

    case ("pow", Array(a, b)) =>
      AST.evalComposeNumeric[Double, Double](ec, a, b)((b, x) => math.pow(b, x))
    case ("log", Array(a, b)) =>
      AST.evalComposeNumeric[Double, Double](ec, a, b)((x, b) => math.log(x) / math.log(b))

    case (_, _) => FunctionRegistry.lookupFun(ec)(fn, args.map(_.`type`).toSeq)(args)
      .getOrElse(fatal(s"No function found with name `$fn' and argument ${ plural(args.size, "type") } `${ args.map(_.`type`).mkString(", ") }'"))
  }

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

      case (TString, "replace", rhs) => {
        lhs.typecheck(ec)
        rhs.foreach(_.typecheck(ec))
        rhs.map(_.`type`) match {
          case Array(TString, TString) => TString
          case other =>
            val nArgs = other.length
            parseError(
              s"""method `$method' expects 2 arguments of type String, e.g. str.replace(" ", "_")
                  |  Found $nArgs ${ plural(nArgs, "argument") }${
                if (nArgs > 0)
                  s"of ${ plural(nArgs, "type") } [${ other.mkString(", ") }]"
                else ""
              }""".stripMargin)
        }
      }

      case (it: TIterable, "find", rhs) =>
        val (param, body) = rhs match {
          case Array(Lambda(_, p, b)) => (p, b)
          case _ => parseError(s"method `$method' expects a lambda function (param => Boolean), " +
            s"e.g. `x => x < 5' or `tc => tc.canonical == 1'")
        }
        body.typecheck(ec.copy(st = ec.st + ((param, (-1, it.elementType)))))
        if (body.`type` != TBoolean)
          parseError(s"method `$method' expects a lambda function (param => Boolean), got (param => ${ body.`type` })")
        `type` = it.elementType

      case (it: TIterable, "map", rhs) =>
        val (param, body) = rhs match {
          case Array(Lambda(_, p, b)) => (p, b)
          case _ => parseError(s"method `$method' expects a lambda function (param => Any), " +
            s"e.g. `x => x * 10' or `tc => tc.gene_symbol'")
        }
        body.typecheck(ec.copy(st = ec.st + ((param, (-1, it.elementType)))))
        `type` = body.`type` match {
          case t: Type => it match {
            case TArray(_) => TArray(t)
            case TSet(_) => TSet(t)
          }
          case error =>
            parseError(s"method `$method' expects a lambda function (param => Any), got invalid mapping (param => $error)")
        }

      case (agg: TAggregable, "map", rhs) =>
        val (param, body) = rhs match {
          case Array(Lambda(_, p, b)) => (p, b)
          case _ => parseError(s"method `$method' expects a lambda function (param => Any), " +
            s"e.g. `x => x < 5' or `tc => tc.canonical'")
        }

        val localIdx = agg.ec.a.length
        val localA = agg.ec.a
        localA += null
        val st = agg.ec.st + ((param, (localIdx, agg.elementType)))
        body.typecheck(agg.ec.copy(st = st))
        `type` = body.`type` match {
          case t: Type =>
            val fn = body.eval(agg.ec.copy(st = st))
            val mapF = (a: Any) => {
              localA(localIdx) = a
              fn()
            }
            MappedAggregable(agg, t, mapF)
          case error =>
            parseError(s"method `$method' expects a lambda function (param => Any), got invalid mapping (param => $error)")
        }

      case (agg: TAggregable, "filter", rhs) =>
        val (param, body) = rhs match {
          case Array(Lambda(_, p, b)) => (p, b)
          case _ => parseError(s"method `$method' expects a lambda function (param => Boolean), " +
            s"e.g. `x => x < 5' or `tc => tc.canonical == 1'")
        }

        val localIdx = agg.ec.a.length
        val localA = agg.ec.a
        localA += null
        val st = agg.ec.st + ((param, (localIdx, agg.elementType)))
        body.typecheck(agg.ec.copy(st = st))
        `type` = body.`type` match {
          case TBoolean =>
            val fn = body.eval(agg.ec.copy(st = st))
            val filterF = (a: Any) => {
              localA(localIdx) = a
              fn().asInstanceOf[Boolean]
            }
            FilteredAggregable(agg, filterF)
          case error =>
            parseError(s"method `$method' expects a lambda function (param => Boolean), but found (param => $error)")
        }

      case (it: TIterable, "flatMap", rhs) =>
        lhs.typecheck(ec)
        val (param, body) = rhs match {
          case Array(Lambda(_, p, b)) => (p, b)
          case _ => parseError(s"method `$method' expects a lambda function [param => Any]")
        }
        body.typecheck(ec.copy(st = ec.st + ((param, (-1, it.elementType)))))
        `type` = body.`type` match {
          case t: TIterable => (t, it) match {
            case (TArray(et), TArray(_)) => TArray(et)
            case (TSet(et), TSet(_)) => TSet(et)
            case _ =>
              parseError(s"method `$method' expects the collection types of the left operand and the lambda body to match, " +
                s"got $it and $t. Consider converting one collection type to the other")
          }
          case error =>
            parseError(s"method `$method' expects lambda body to have type Array[T] or Set[T], got $error")
        }

      case (it: TIterable, "flatten", rhs) =>
        lhs.typecheck(ec)
        if (!rhs.isEmpty)
          parseError(s"method `$method' does not take parameters, use flatten()")
        `type` = it match {
          case TArray(TArray(e)) => TArray(e)
          case TSet(TSet(e)) => TSet(e)
          case _ => parseError(s"method `$method' expects type Array[Array[T]] or Set[Set[T]], got $it.")
        }

      case (it: TIterable, "filter", rhs) =>
        val (param, body) = rhs match {
          case Array(Lambda(_, p, b)) => (p, b)
          case _ => parseError(s"method `$method' expects a lambda function (param => Boolean), " +
            s"e.g. `x => x < 5' or `tc => tc.canonical == 1'")
        }
        body.typecheck(ec.copy(st = ec.st + ((param, (-1, it.elementType)))))
        if (body.`type` != TBoolean)
          parseError(s"method `$method' expects a lambda function (param => Boolean), got (param => ${ body.`type` })")
        `type` = it

      case (it: TIterable, "forall" | "exists", rhs) =>
        val (param, body) = rhs match {
          case Array(Lambda(_, p, b)) => (p, b)
          case _ => parseError(s"method `$method' expects a lambda function (param => Boolean), " +
            s"e.g. `x => x < 5' or `tc => tc.canonical == 1'")
        }
        body.typecheck(ec.copy(st = ec.st + ((param, (-1, it.elementType)))))
        if (body.`type` != TBoolean)
          parseError(s"method `$method' expects a lambda function (param => Boolean), got (param => ${ body.`type` })")
        `type` = TBoolean

      case (arr: TArray, "sort", rhs) =>
        rhs match {
          case Array() =>
          case Array(Const(_, _, TBoolean)) =>
          case _ => parseError(s"method `$method' expects at most one Boolean parameter")
        }
        `type` = arr

      case (arr: TArray, "sortBy", rhs) =>
        val (param, body) = rhs match {
          case Array(Lambda(_, p, b)) => (p, b)
          case Array(Lambda(_, p, b), Const(_, _, TBoolean)) => (p, b)
          case _ => parseError(s"method `$method' expects a lambda function (param => T) and at most one Boolean parameter")
        }
        body.typecheck(ec.copy(st = ec.st + ((param, (-1, arr.elementType)))))
        if (!(body.`type`.isInstanceOf[TNumeric] || body.`type` == TString))
          parseError(s"method `$method' expects a lambda function (param => T) with T of string or numeric type, got (param => ${ body.`type` })")
        `type` = arr

      case (TDict(elementType), "mapValues", rhs) =>
        lhs.typecheck(ec)
        val (param, body) = rhs match {
          case Array(Lambda(_, p, b)) => (p, b)
          case _ => parseError(s"method `$method' expects a lambda function (param => Any), " +
            s"e.g. `x => x < 5' or `tc => tc.canonical'")
        }
        body.typecheck(ec.copy(st = ec.st + ((param, (-1, elementType)))))
        `type` = body.`type` match {
          case t: Type => TDict(t)
          case error =>
            parseError(s"method `$method' expects a lambda function (param => Any), got invalid mapping (param => $error)")
        }

      case (agg: TAggregable, "count", rhs) =>
        if (rhs.nonEmpty)
          parseError(s"""method `$method' does not take arguments""")
        `type` = TLong

      case (agg: TAggregable, "fraction", rhs) =>
        val (param, body) = rhs match {
          case Array(Lambda(_, p, b)) => (p, b)
          case _ => parseError(s"method `$method' expects a lambda function (param => Boolean), " +
            s"e.g. `g => g.gq < 5' or `x => x == 2'")
        }
        body.typecheck(ec.copy(st = ec.st + ((param, (-1, agg.elementType)))))

        `type` = TDouble

      case (agg: TAggregable, "stats", rhs) =>
        if (rhs.nonEmpty)
          parseError(s"""method `$method' does not take arguments""")
        if (!agg.elementType.isInstanceOf[TNumeric])
          parseError(
            s"""method `$method' can only operate on Aggregable[Numeric]
                |  Found `$agg'""".stripMargin)

        `type` = TStruct(("mean", TDouble), ("stdev", TDouble), ("min", TDouble),
          ("max", TDouble), ("nNotMissing", TLong), ("sum", TDouble))

      case (agg: TAggregable, "hist", rhs) =>
        rhs match {
          case Array(startAST, endAST, binsAST) =>

            rhs.foreach(_.typecheck(ec))
            rhs.foreach(_.errorIf[SymRef]("method `hist' cannot contain variable references"))

            val types = rhs.map(_.`type`)
            types match {
              case Array(_: TNumeric, _: TNumeric, TInt) =>
              case _ => parseError(
                s"""method `hist' expects arguments of type (Numeric, Numeric, Int)
                    |  Found ${ types.mkString(", ") }""".stripMargin)
            }

            `type` = HistogramCombiner.schema

          case _ => parseError(
            s"""method `hist' expects three numeric arguments (start, end, bins)
                |  Examples:
                |    gs.map(g => g.gq).hist(0, 100, 20)
                |    variants.map(v => va.linreg.beta).hist(.5, 1.5, 100)
             """.stripMargin)
        }

      case (agg: TAggregable, "collect", rhs) =>
        if (rhs.nonEmpty)
          parseError(s"""method `$method' does not take arguments""")
        `type` = TArray(agg.elementType)

      case (agg: TAggregable, "sum", rhs) =>
        if (rhs.nonEmpty)
          parseError(s"""method `$method' does not take arguments""")
        `type` = agg.elementType match {
          case _: TNumeric => TDouble
          case TArray(_: TNumeric) => TArray(TDouble)
          case _ => parseError(
            s"""method `$method' can not operate on `$agg'
                |  Accepted aggregable types: `Aggregable[Numeric]' and `Aggregable[Array[Numeric]]'
                |  Hint: use `.map(x => ...)' to produce a numeric aggregable""".stripMargin)
        }
      case (agg: TAggregable, "infoScore", rhs) =>
        if (rhs.nonEmpty)
          parseError(s"""method `$method' does not take arguments""")
        `type` = agg.elementType match {
          case TGenotype => InfoScoreCombiner.signature
          case _ => parseError(
            s"""method `$method' can not operate on `$agg'
                |  Accepted aggregable type: `Aggregable[Genotype]'
             """.stripMargin
          )
        }
      case _ =>
        super.typecheck(ec)
    }
  }

  override def typecheckThis(): BaseType = {
    val rhsTypes = args.map(_.`type`)
    (lhs.`type`, method, rhsTypes) match {
      case (TArray(TString), "mkString", Array(TString)) => TString
      case (TSet(elementType), "contains", Array(t2)) =>
        if (elementType != t2)
          parseError(
            s"""method `contains' takes an argument of the same type as the set.
                |  Expected type `$elementType' for `Set[$elementType]', but found `$t2'""".stripMargin)
        TBoolean
      case (TDict(_), "contains", Array(TString)) => TBoolean
      case (TInterval, "contains", Array(TLocus)) => TBoolean
      case (TString, "split", Array(TString)) => TArray(TString)

      case (t: TNumeric, "min", Array(t2: TNumeric)) =>
        AST.promoteNumeric(t, t2)
      case (t: TNumeric, "max", Array(t2: TNumeric)) =>
        AST.promoteNumeric(t, t2)

      case (t, "orElse", Array(t2)) if t == t2 =>
        t

      case (TGenotype, "oneHotAlleles", Array(TVariant)) => TArray(TInt)

      case (t, _, _) =>
        parseError(s"`no matching signature for `$method(${ rhsTypes.mkString(", ") })' on `$t'")
    }
  }

  def eval(ec: EvalContext): () => Any = ((lhs.`type`, method, args): @unchecked) match {
    case (returnType, "find", Array(Lambda(_, param, body))) =>
      val localIdx = ec.a.length
      val localA = ec.a
      localA += null
      val bodyFn = body.eval(ec.copy(st = ec.st + (param -> (localIdx, returnType))))

      AST.evalCompose[Iterable[_]](ec, lhs) { s =>
        s.find { elt =>
          localA(localIdx) = elt
          val r = bodyFn()
          r != null && r.asInstanceOf[Boolean]
        }.orNull
      }

    case (it: TIterable, "map", Array(Lambda(_, param, body))) =>
      val localIdx = ec.a.length
      val localA = ec.a
      localA += null
      val bodyFn = body.eval(ec.copy(st = ec.st + (param -> (localIdx, it))))

      (it: @unchecked) match {
        case TArray(_) =>
          AST.evalCompose[IndexedSeq[_]](ec, lhs) { is =>
            is.map { elt =>
              localA(localIdx) = elt
              bodyFn()
            }
          }
        case TSet(_) =>
          AST.evalCompose[Set[_]](ec, lhs) { s =>
            s.map { elt =>
              localA(localIdx) = elt
              bodyFn()
            }
          }
      }

    case (it: TIterable, "flatMap", Array(Lambda(_, param, body))) =>
      val localIdx = ec.a.length
      val localA = ec.a
      localA += null
      val bodyFn = body.eval(ec.copy(st = ec.st + (param -> (localIdx, it))))

      (it: @unchecked) match {
        case TArray(_) =>
          AST.evalCompose[IndexedSeq[_]](ec, lhs) { is =>
            flattenOrNull[IndexedSeq, Any](IndexedSeq.newBuilder[Any],
              is.map { elt =>
                localA(localIdx) = elt
                bodyFn().asInstanceOf[Iterable[_]]
              }
            )
          }
        case TSet(_) =>
          AST.evalCompose[Set[_]](ec, lhs) { s =>
            flattenOrNull[Set, Any](Set.newBuilder[Any],
              s.map { elt =>
                localA(localIdx) = elt
                bodyFn().asInstanceOf[Iterable[_]]
              }
            )
          }
      }

    case (it: TIterable, "flatten", Array()) =>
      (it: @unchecked) match {
        case TArray(_) =>
          AST.evalCompose[IndexedSeq[_]](ec, lhs) { is =>
            flattenOrNull[IndexedSeq, Any](IndexedSeq.newBuilder[Any], is.asInstanceOf[Iterable[Iterable[_]]])
          }
        case TSet(_) =>
          AST.evalCompose[Set[_]](ec, lhs) { s =>
            flattenOrNull[Set, Any](Set.newBuilder[Any], s.asInstanceOf[Iterable[Iterable[_]]])
          }
      }

    case (it: TIterable, "filter", Array(Lambda(_, param, body))) =>
      val localIdx = ec.a.length
      val localA = ec.a
      localA += null
      val bodyFn = body.eval(ec.copy(st = ec.st + (param -> (localIdx, it))))

      (it: @unchecked) match {
        case TArray(_) =>
          AST.evalCompose[IndexedSeq[_]](ec, lhs) { is =>
            is.filter { elt =>
              localA(localIdx) = elt
              val r = bodyFn()
              r.asInstanceOf[Boolean]
            }
          }
        case TSet(_) =>
          AST.evalCompose[Set[_]](ec, lhs) { s =>
            s.filter { elt =>
              localA(localIdx) = elt
              val r = bodyFn()
              r.asInstanceOf[Boolean]
            }
          }
      }

    case (returnType, "forall", Array(Lambda(_, param, body))) =>
      val localIdx = ec.a.length
      val localA = ec.a
      localA += null
      val bodyFn = body.eval(ec.copy(st = ec.st + (param -> (localIdx, returnType))))

      AST.evalCompose[Iterable[_]](ec, lhs) { is =>
        is.forall { elt =>
          localA(localIdx) = elt
          val r = bodyFn()
          r.asInstanceOf[Boolean]
        }
      }

    case (returnType, "exists", Array(Lambda(_, param, body))) =>
      val localIdx = ec.a.length
      val localA = ec.a
      localA += null
      val bodyFn = body.eval(ec.copy(st = ec.st + (param -> (localIdx, returnType))))

      AST.evalCompose[Iterable[_]](ec, lhs) { is =>
        is.exists { elt =>
          localA(localIdx) = elt
          val r = bodyFn()
          r.asInstanceOf[Boolean]
        }
      }

    case (returnType, "sort", rhs) =>
      val ascending = (rhs: @unchecked) match {
        case Array() => true
        case Array(Const(_, asc, _)) => asc.asInstanceOf[Boolean]
      }
      val baseOrd = (returnType: @unchecked) match {
        case TArray(TDouble) => Ordering.Double
        case TArray(TFloat) => Ordering.Float
        case TArray(TLong) => Ordering.Long
        case TArray(TInt) => Ordering.Int
        case TArray(TString) => Ordering.String
      }
      val ord = extendOrderingToNull(
        if (ascending)
          baseOrd
        else
          baseOrd.reverse
      )
      AST.evalCompose[IndexedSeq[_]](ec, lhs) { arr =>
        arr.sorted(ord)
      }

    case (returnType, "sortBy", rhs) =>
      val ascending = (rhs: @unchecked) match {
        case Array(_) => true
        case Array(_, Const(_, asc, _)) => asc.asInstanceOf[Boolean]
      }
      val Lambda(_, param, body) = rhs(0)
      val baseOrd = (body.`type`: @unchecked) match {
        case TDouble => Ordering.Double
        case TFloat => Ordering.Float
        case TLong => Ordering.Long
        case TInt => Ordering.Int
        case TString => Ordering.String
      }
      val ord = extendOrderingToNull(
        if (ascending)
          baseOrd
        else
          baseOrd.reverse)
      val localIdx = ec.a.length
      val localA = ec.a
      localA += null
      val bodyFn = body.eval(ec.copy(st = ec.st + (param -> (localIdx, returnType))))
      AST.evalCompose[IndexedSeq[_]](ec, lhs) { arr =>
        arr.sortBy { elt =>
          localA(localIdx) = elt
          bodyFn()
        }(ord)
      }

    case (returnType, "mapValues", Array(Lambda(_, param, body))) =>
      val localIdx = ec.a.length
      val localA = ec.a
      localA += null
      val bodyFn = body.eval(ec.copy(st = ec.st + (param -> (localIdx, returnType))))

      AST.evalCompose[Map[_, _]](ec, lhs) { m =>
        m.mapValues { elt =>
          localA(localIdx) = elt
          bodyFn()
        }.force
      }

    case (agg: TAggregable, "count", Array()) =>
      val localA = agg.ec.a
      val localIdx = localA.length
      localA += null

      val aggF = agg.f
      agg.ec.aggregationFunctions += new TypedAggregator[Long] {
        override def zero: Long = 0L

        override def seqOp(x: Any, acc: Long): Long = {
          aggF(x).map(_ => acc + 1).getOrElse(acc)
        }

        override def combOp(acc1: Long, acc2: Long): Long = acc1 + acc2

        override def idx = localIdx
      }.erase

      () => localA(localIdx)

    case (agg: TAggregable, "fraction", Array(Lambda(_, param, body))) =>
      val localA = agg.ec.a
      val localIdx = localA.length
      localA += null

      val lambdaIdx = localA.length
      localA += null
      val bodyFn = body.eval(agg.ec.copy(st = agg.ec.st + (param -> (lambdaIdx, agg.elementType))))

      val aggF = agg.f
      agg.ec.aggregationFunctions += new TypedAggregator[(Long, Long)] {
        override def zero: (Long, Long) = (0L, 0L)

        override def seqOp(x: Any, acc: (Long, Long)): (Long, Long) = {
          val (numerator, denominator) = acc
          aggF(x).map { value =>
            localA(lambdaIdx) = value
            val numToAdd = if (bodyFn().asInstanceOf[Boolean]) 1 else 0
            (numerator + numToAdd, denominator + 1)
          }.getOrElse((numerator, denominator))
        }

        override def combOp(acc1: (Long, Long), acc2: (Long, Long)): (Long, Long) = (acc1._1 + acc2._1, acc1._2 + acc2._2)

        override def idx = localIdx
      }.erase

      () => {
        val (num: Long, denom: Long) = localA(localIdx)
        divNull(num.toDouble, denom)
      }

    case (agg: TAggregable, "stats", Array()) =>
      val localA = agg.ec.a
      val localIdx = localA.length
      localA += null

      val t = agg.elementType
      val aggF = agg.f

      agg.ec.aggregationFunctions += new TypedAggregator[StatCounter] {
        override def zero: StatCounter = new StatCounter()

        override def seqOp(x: Any, acc: StatCounter): StatCounter = {
          aggF(x).foreach(x => acc.merge(DoubleNumericConversion.to(x)))
          acc
        }

        override def combOp(acc1: StatCounter, acc2: StatCounter): StatCounter = acc1.merge(acc2)

        override def idx = localIdx
      }.erase

      val getOp = (a: Any) => {
        val sc = a.asInstanceOf[StatCounter]
        if (sc.count == 0)
          null
        else
          Annotation(sc.mean, sc.stdev, sc.min, sc.max, sc.count, sc.sum)
      }

      () => {
        val sc = localA(localIdx).asInstanceOf[StatCounter]
        if (sc.count == 0)
          null
        else
          Annotation(sc.mean, sc.stdev, sc.min, sc.max, sc.count, sc.sum)
      }

    case (agg: TAggregable, "hist", Array(startAST, endAST, binsAST)) =>

      val start = DoubleNumericConversion.to(startAST.eval(ec)())
      val end = DoubleNumericConversion.to(endAST.eval(ec)())
      val bins = binsAST.eval(ec)().asInstanceOf[Int]

      if (bins <= 0)
        parseError(s"""method `hist' expects `bins' argument to be > 0, but got $bins""")

      val binSize = (end - start) / bins
      if (binSize <= 0)
        parseError(
          s"""invalid bin size from given arguments (start = $start, end = $end, bins = $bins)
              |  Method requires positive bin size [(end - start) / bins], but got ${ binSize.formatted("%.2f") }
                 """.stripMargin)

      val indices = Array.tabulate(bins + 1)(i => start + i * binSize)

      info(
        s"""computing histogram with the following bins:
            |  ${
          val fString = math.log10(indices.length - 1).ceil.toInt
          val longestBound = indices.map(_.formatted("%.2f").length).max
          def formatRange(d: Double): String = d.formatted("%.2f")
          val maxStrLength = indices.map(formatRange).map(_.length).max
          val formatted = indices.map(formatRange).map(_.formatted(s"%${ maxStrLength }s"))

          formatted.zip(formatted.tail)
            .zipWithIndex
            .map { case ((l, r), index) =>
              val rightBound = if (index == indices.length - 2) "]" else ")"
              s"bin ${ index.formatted(s"%0${ fString }d") }: [$l, $r$rightBound"
            }
            .grouped(2)
            .map { arr => arr.mkString(",   ")
            }.mkString(",\n  ")
        }""".stripMargin)

      val localIdx = agg.ec.a.length
      val localA = agg.ec.a
      localA += null

      val aggF = agg.f
      agg.ec.aggregationFunctions += new TypedAggregator[HistogramCombiner] {
        override def zero: HistogramCombiner = new HistogramCombiner(indices)

        override def seqOp(x: Any, acc: HistogramCombiner): HistogramCombiner = {
          aggF(x).foreach(x => acc.merge(DoubleNumericConversion.to(x)))
          acc
        }

        override def combOp(acc1: HistogramCombiner, acc2: HistogramCombiner): HistogramCombiner = acc1.merge(acc2)

        override def idx = localIdx
      }.erase

      () => localA(localIdx).asInstanceOf[HistogramCombiner].toAnnotation


    case (agg: TAggregable, "collect", Array()) =>
      val localIdx = agg.ec.a.length
      val localA = agg.ec.a
      localA += null

      val aggF = agg.f
      agg.ec.aggregationFunctions += new TypedAggregator[ArrayBuffer[Any]] {
        override def zero: ArrayBuffer[Any] = new ArrayBuffer[Any]

        override def seqOp(x: Any, acc: ArrayBuffer[Any]): ArrayBuffer[Any] = {
          aggF(x).foreach(elem => acc += elem)
          acc
        }

        override def combOp(acc1: ArrayBuffer[Any], acc2: ArrayBuffer[Any]): ArrayBuffer[Any] = {
          acc1 ++= acc2
          acc1
        }

        override def idx = localIdx
      }.erase

      () => localA(localIdx).asInstanceOf[ArrayBuffer[Any]].toIndexedSeq

    case (agg: TAggregable, "infoScore", Array()) =>
      val localIdx = agg.ec.a.length
      val localA = agg.ec.a
      localA += null

      val localPos = posn
      val aggF = agg.f
      agg.ec.aggregationFunctions += new TypedAggregator[InfoScoreCombiner] {

        override def zero: InfoScoreCombiner = new InfoScoreCombiner()

        override def seqOp(x: Any, acc: InfoScoreCombiner): InfoScoreCombiner = {
          aggF(x).foreach(x => acc.merge(x.asInstanceOf[Genotype]))
          acc
        }

        override def combOp(acc1: InfoScoreCombiner, acc2: InfoScoreCombiner): InfoScoreCombiner = acc1.merge(acc2)

        override def idx = localIdx
      }.erase

      () => localA(localIdx).asInstanceOf[InfoScoreCombiner].asAnnotation

    case (agg: TAggregable, "sum", Array()) =>
      val localIdx = agg.ec.a.length
      val localA = agg.ec.a
      localA += null

      val localPos = posn
      val aggF = agg.f
      (`type`: @unchecked) match {
        case TDouble => agg.ec.aggregationFunctions += new TypedAggregator[Double] {
          override def zero: Double = 0d

          override def seqOp(x: Any, acc: Double): Double = aggF(x)
            .map(elem => DoubleNumericConversion.to(elem) + acc)
            .getOrElse(acc)

          override def combOp(acc1: Double, acc2: Double): Double = acc1 + acc2

          override def idx: Int = localIdx
        }.erase

        case TArray(TDouble) => agg.ec.aggregationFunctions += new TypedAggregator[IndexedSeq[Double]] {
          override def zero: IndexedSeq[Double] = null

          override def seqOp(x: Any, acc: IndexedSeq[Double]): IndexedSeq[Double] = aggF(x)
            .map(_.asInstanceOf[IndexedSeq[_]])
            .map { arr =>
              val cast = arr.map(a => if (a == null) 0d else DoubleNumericConversion.to(a))
              if (acc == null)
                cast
              else {
                if (acc.length != cast.length)
                  ParserUtils.error(localPos,
                    s"""cannot aggregate arrays of unequal length with `sum'
                        |  Found conflicting arrays of size (${ acc.size }) and (${ cast.size })""".stripMargin)
                (acc, cast).zipped.map(_ + _)
              }
            }.getOrElse(acc)

          override def combOp(acc1: IndexedSeq[Double], acc2: IndexedSeq[Double]): IndexedSeq[Double] = {
            if (acc1.length != acc2.length)
              ParserUtils.error(localPos,
                s"""cannot aggregate arrays of unequal length with `sum'
                    |  Found conflicting arrays of size (${ acc1.size }) and (${ acc2.size })""".stripMargin)
            (acc1, acc2).zipped.map(_ + _)
          }

          override def idx: Int = localIdx
        }.erase
      }

      () => localA(localIdx)

    case (_, "orElse", Array(a)) =>
      val f1 = lhs.eval(ec)
      val f2 = a.eval(ec)
      () => {
        val v = f1()
        if (v == null)
          f2()
        else
          v
      }

    case (TGenotype, "oneHotAlleles", Array(v)) =>
      AST.evalCompose[Genotype, Variant](ec, lhs, v) { case (g, v) => g.oneHotAlleles(v).map(_.toIndexedSeq).orNull }

    case (TString, "replace", Array(a, b)) =>
      AST.evalCompose[String, String, String](ec, lhs, a, b) { case (str, pattern1, pattern2) =>
        str.replaceAll(pattern1, pattern2)
      }

    case (TArray(elementType), "mkString", Array(a)) =>
      AST.evalCompose[IndexedSeq[String], String](ec, lhs, a) { case (s, t) => s.map(elementType.str).mkString(t) }
    case (TSet(elementType), "contains", Array(a)) =>
      AST.evalCompose[Set[Any], Any](ec, lhs, a) { case (a, x) => a.contains(x) }
    case (TSet(elementType), "mkString", Array(a)) =>
      AST.evalCompose[IndexedSeq[String], String](ec, lhs, a) { case (s, t) => s.map(elementType.str).mkString(t) }

    case (TDict(elementType), "contains", Array(a)) =>
      AST.evalCompose[Map[String, _], String](ec, lhs, a) { case (m, key) => m.contains(key) }

    case (TInterval, "contains", Array(l)) =>
      AST.evalCompose[Interval[Locus], Locus](ec, lhs, l) { case (interval, locus) => interval.contains(locus) }

    case (TString, "split", Array(a)) =>
      AST.evalCompose[String, String](ec, lhs, a) { case (s, p) => s.split(p): IndexedSeq[String] }

    case (TInt, "min", Array(a)) => AST.evalComposeNumeric[Int, Int](ec, lhs, a)(_ min _)
    case (TLong, "min", Array(a)) => AST.evalComposeNumeric[Long, Long](ec, lhs, a)(_ min _)
    case (TFloat, "min", Array(a)) => AST.evalComposeNumeric[Float, Float](ec, lhs, a)(_ min _)
    case (TDouble, "min", Array(a)) => AST.evalComposeNumeric[Double, Double](ec, lhs, a)(_ min _)

    case (TInt, "max", Array(a)) => AST.evalComposeNumeric[Int, Int](ec, lhs, a)(_ max _)
    case (TLong, "max", Array(a)) => AST.evalComposeNumeric[Long, Long](ec, lhs, a)(_ max _)
    case (TFloat, "max", Array(a)) => AST.evalComposeNumeric[Float, Float](ec, lhs, a)(_ max _)
    case (TDouble, "max", Array(a)) => AST.evalComposeNumeric[Double, Double](ec, lhs, a)(_ max _)
  }

}

case class Let(posn: Position, bindings: Array[(String, AST)], body: AST) extends AST(posn, bindings.map(_._2) :+ body) {

  def eval(ec: EvalContext): () => Any = {
    val indexb = new mutable.ArrayBuilder.ofInt
    val bindingfb = mutable.ArrayBuilder.make[() => Any]()

    var symTab2 = ec.st
    val localA = ec.a
    for ((id, v) <- bindings) {
      val i = localA.length
      localA += null
      bindingfb += v.eval(ec.copy(st = symTab2))
      indexb += i
      symTab2 = symTab2 + (id -> (i, v.`type`))
    }

    val n = bindings.length
    val indices = indexb.result()
    val bindingfs = bindingfb.result()
    val bodyf = body.eval(ec.copy(st = symTab2))
    () => {
      for (i <- 0 until n)
        localA(indices(i)) = bindingfs(i)()
      bodyf()
    }
  }

  override def typecheck(ec: EvalContext) {
    var symTab2 = ec.st
    for ((id, v) <- bindings) {
      v.typecheck(ec.copy(st = symTab2))
      symTab2 = symTab2 + (id -> (-1, v.`type`))
    }
    body.typecheck(ec.copy(st = symTab2))

    `type` = body.`type`
  }
}

case class BinaryOp(posn: Position, lhs: AST, operation: String, rhs: AST) extends AST(posn, lhs, rhs) {
  def eval(ec: EvalContext): () => Any = ((operation, `type`): @unchecked) match {
    case ("+", TString) =>
      val lhsT = lhs.`type`.asInstanceOf[Type]
      val rhsT = rhs.`type`.asInstanceOf[Type]
      AST.evalCompose[Any, Any](ec, lhs, rhs) { (left, right) => lhsT.str(left) + rhsT.str(right) }
    case ("~", TBoolean) => AST.evalCompose[String, String](ec, lhs, rhs) { (s, t) =>
      s.r.findFirstIn(t).isDefined
    }

    case ("+" | "*" | "-" | "/", TArray(elementType)) =>
      val f: (Any, Any) => Any = (operation, elementType) match {
        case ("+", TDouble) => AST.evalNumeric[Double, Double](_ + _)
        case ("+", TInt) => AST.evalNumeric[Int, Int](_ + _)
        case ("+", TLong) => AST.evalNumeric[Long, Long](_ + _)
        case ("+", TFloat) => AST.evalNumeric[Float, Float](_ + _)
        case ("*", TDouble) => AST.evalNumeric[Double, Double](_ * _)
        case ("*", TInt) => AST.evalNumeric[Int, Int](_ * _)
        case ("*", TLong) => AST.evalNumeric[Long, Long](_ * _)
        case ("*", TFloat) => AST.evalNumeric[Float, Float](_ * _)
        case ("-", TDouble) => AST.evalNumeric[Double, Double](_ - _)
        case ("-", TInt) => AST.evalNumeric[Int, Int](_ - _)
        case ("-", TLong) => AST.evalNumeric[Long, Long](_ - _)
        case ("-", TFloat) => AST.evalNumeric[Float, Float](_ - _)
        case ("/", TDouble) => AST.evalNumeric[Double, Double](_ / _)
      }

      ((lhs.`type`, rhs.`type`): @unchecked) match {
        case (TArray(_), TArray(_)) => AST.evalCompose[IndexedSeq[_], IndexedSeq[_]](ec, lhs, rhs) { case (left, right) =>
          if (left.length != right.length) ParserUtils.error(posn,
            s"""cannot apply operation `$operation' to arrays of unequal length
                |  Left: ${ left.length } elements
                |  Right: ${ right.length } elements""".stripMargin)
          (left, right).zipped.map(f)
        }
        case (_, TArray(_)) => AST.evalCompose[Any, IndexedSeq[_]](ec, lhs, rhs) { case (num, arr) =>
          arr.map(elem => f(num, elem))
        }
        case (TArray(_), _) => AST.evalCompose[IndexedSeq[_], Any](ec, lhs, rhs) { case (arr, num) =>
          arr.map(elem => f(elem, num))
        }
      }

    case ("||", TBoolean) =>
      val f1 = lhs.eval(ec)
      val f2 = rhs.eval(ec)
      () => {
        val x1 = f1()
        if (x1 != null) {
          if (x1.asInstanceOf[Boolean])
            true
          else
            f2()
        } else {
          val x2 = f2()
          if (x2 != null
            && x2.asInstanceOf[Boolean])
            true
          else
            null
        }
      }

    case ("&&", TBoolean) =>
      val f1 = lhs.eval(ec)
      val f2 = rhs.eval(ec)
      () => {
        val x = f1()
        if (x != null) {
          if (x.asInstanceOf[Boolean])
            f2()
          else
            false
        } else {
          val x2 = f2()
          if (x2 != null
            && !x2.asInstanceOf[Boolean])
            false
          else
            null
        }
      }

    case ("+", TInt) => AST.evalComposeNumeric[Int, Int](ec, lhs, rhs)(_ + _)
    case ("-", TInt) => AST.evalComposeNumeric[Int, Int](ec, lhs, rhs)(_ - _)
    case ("*", TInt) => AST.evalComposeNumeric[Int, Int](ec, lhs, rhs)(_ * _)
    case ("/", TInt) => AST.evalComposeNumeric[Double, Double](ec, lhs, rhs)(_ / _)
    case ("%", TInt) => AST.evalComposeNumeric[Int, Int](ec, lhs, rhs)(_ % _)

    case ("+", TLong) => AST.evalComposeNumeric[Long, Long](ec, lhs, rhs)(_ + _)
    case ("-", TLong) => AST.evalComposeNumeric[Long, Long](ec, lhs, rhs)(_ - _)
    case ("*", TLong) => AST.evalComposeNumeric[Long, Long](ec, lhs, rhs)(_ * _)
    case ("/", TLong) => AST.evalComposeNumeric[Double, Double](ec, lhs, rhs)(_ / _)
    case ("%", TLong) => AST.evalComposeNumeric[Long, Long](ec, lhs, rhs)(_ % _)

    case ("+", TFloat) => AST.evalComposeNumeric[Float, Float](ec, lhs, rhs)(_ + _)
    case ("-", TFloat) => AST.evalComposeNumeric[Float, Float](ec, lhs, rhs)(_ - _)
    case ("*", TFloat) => AST.evalComposeNumeric[Float, Float](ec, lhs, rhs)(_ * _)
    case ("/", TFloat) => AST.evalComposeNumeric[Float, Float](ec, lhs, rhs)(_ / _)

    case ("+", TDouble) => AST.evalComposeNumeric[Double, Double](ec, lhs, rhs)(_ + _)
    case ("-", TDouble) => AST.evalComposeNumeric[Double, Double](ec, lhs, rhs)(_ - _)
    case ("*", TDouble) => AST.evalComposeNumeric[Double, Double](ec, lhs, rhs)(_ * _)
    case ("/", TDouble) => AST.evalComposeNumeric[Double, Double](ec, lhs, rhs)(_ / _)

  }

  override def typecheckThis(): BaseType = (lhs.`type`, operation, rhs.`type`) match {
    case (t: Type, "+", TString) => TString
    case (TString, "+", t: Type) => TString
    case (TString, "~", TString) => TBoolean
    case (TBoolean, "||", TBoolean) => TBoolean
    case (TBoolean, "&&", TBoolean) => TBoolean
    case (lhsType: TIntegral, "%", rhsType: TIntegral) => AST.promoteNumeric(lhsType, rhsType)
    case (lhsType: TNumeric, "+", rhsType: TNumeric) => AST.promoteNumeric(lhsType, rhsType)
    case (lhsType: TNumeric, "-", rhsType: TNumeric) => AST.promoteNumeric(lhsType, rhsType)
    case (lhsType: TNumeric, "*", rhsType: TNumeric) => AST.promoteNumeric(lhsType, rhsType)
    case (lhsType: TNumeric, "/", rhsType: TNumeric) => TDouble

    case (TArray(lhsType: TNumeric), "+" | "-" | "*", TArray(rhsType: TNumeric)) =>
      TArray(AST.promoteNumeric(lhsType, rhsType))
    case (TArray(lhsType: TNumeric), "/", TArray(rhsType: TNumeric)) => TArray(TDouble)

    case (lhsType: TNumeric, "+" | "-" | "*", TArray(rhsType: TNumeric)) =>
      TArray(AST.promoteNumeric(lhsType, rhsType))
    case (lhsType: TNumeric, "/", TArray(rhsType: TNumeric)) => TArray(TDouble)
    case (TArray(lhsType: TNumeric), "+" | "-" | "*", rhsType: TNumeric) =>
      TArray(AST.promoteNumeric(lhsType, rhsType))
    case (TArray(lhsType: TNumeric), "/", rhsType: TNumeric) => TArray(TDouble)

    case (lhsType, _, rhsType) =>
      parseError(s"invalid arguments to `$operation': ($lhsType, $rhsType)")
  }
}

case class Comparison(posn: Position, lhs: AST, operation: String, rhs: AST) extends AST(posn, lhs, rhs) {
  var operandType: BaseType = null

  def eval(ec: EvalContext): () => Any = ((operation, operandType): @unchecked) match {
    case ("==", _) => AST.evalCompose[Any, Any](ec, lhs, rhs)(_ == _)
    case ("!=", _) => AST.evalCompose[Any, Any](ec, lhs, rhs)(_ != _)

    case ("<", TInt) => AST.evalComposeNumeric[Int, Int](ec, lhs, rhs)(_ < _)
    case ("<=", TInt) => AST.evalComposeNumeric[Int, Int](ec, lhs, rhs)(_ <= _)
    case (">", TInt) => AST.evalComposeNumeric[Int, Int](ec, lhs, rhs)(_ > _)
    case (">=", TInt) => AST.evalComposeNumeric[Int, Int](ec, lhs, rhs)(_ >= _)

    case ("<", TLong) => AST.evalComposeNumeric[Long, Long](ec, lhs, rhs)(_ < _)
    case ("<=", TLong) => AST.evalComposeNumeric[Long, Long](ec, lhs, rhs)(_ <= _)
    case (">", TLong) => AST.evalComposeNumeric[Long, Long](ec, lhs, rhs)(_ > _)
    case (">=", TLong) => AST.evalComposeNumeric[Long, Long](ec, lhs, rhs)(_ >= _)

    case ("<", TFloat) => AST.evalComposeNumeric[Float, Float](ec, lhs, rhs)(_ < _)
    case ("<=", TFloat) => AST.evalComposeNumeric[Float, Float](ec, lhs, rhs)(_ <= _)
    case (">", TFloat) => AST.evalComposeNumeric[Float, Float](ec, lhs, rhs)(_ > _)
    case (">=", TFloat) => AST.evalComposeNumeric[Float, Float](ec, lhs, rhs)(_ >= _)

    case ("<", TDouble) => AST.evalComposeNumeric[Double, Double](ec, lhs, rhs)(_ < _)
    case ("<=", TDouble) => AST.evalComposeNumeric[Double, Double](ec, lhs, rhs)(_ <= _)
    case (">", TDouble) => AST.evalComposeNumeric[Double, Double](ec, lhs, rhs)(_ > _)
    case (">=", TDouble) => AST.evalComposeNumeric[Double, Double](ec, lhs, rhs)(_ >= _)
  }

  override def typecheckThis(): BaseType = {
    operandType = (lhs.`type`, operation, rhs.`type`) match {
      case (lhsType: TNumeric, "==" | "!=" | "<=" | ">=" | "<" | ">", rhsType: TNumeric) =>
        AST.promoteNumeric(lhsType, rhsType)

      case (lhsType, "==" | "!=", rhsType) =>
        if (lhsType != rhsType)
          parseError(s"invalid comparison: `$lhsType' and `$rhsType', can only compare objects of similar type")
        else TBoolean

      case (lhsType, _, rhsType) =>
        parseError(s"invalid arguments to `$operation': ($lhsType, $rhsType)")
    }

    TBoolean
  }
}

case class UnaryOp(posn: Position, operation: String, operand: AST) extends AST(posn, operand) {
  def eval(ec: EvalContext): () => Any = ((operation, `type`): @unchecked) match {
    case ("-", TInt) => AST.evalComposeNumeric[Int](ec, operand)(-_)
    case ("-", TLong) => AST.evalComposeNumeric[Long](ec, operand)(-_)
    case ("-", TFloat) => AST.evalComposeNumeric[Float](ec, operand)(-_)
    case ("-", TDouble) => AST.evalComposeNumeric[Double](ec, operand)(-_)

    case ("!", TBoolean) => AST.evalCompose[Boolean](ec, operand)(!_)
  }

  override def typecheckThis(): BaseType = (operation, operand.`type`) match {
    case ("-", t: TNumeric) => AST.promoteNumeric(t)
    case ("!", TBoolean) => TBoolean

    case (_, t) =>
      parseError(s"invalid argument to unary `$operation': ${ t.toString }")
  }
}

case class IndexOp(posn: Position, f: AST, idx: AST) extends AST(posn, Array(f, idx)) {
  override def typecheckThis(): BaseType = (f.`type`, idx.`type`) match {
    case (TArray(elementType), TInt) => elementType
    case (TDict(elementType), TString) => elementType
    case (TString, TInt) => TChar

    case _ =>
      parseError(
        s""" invalid index expression: cannot index `${ f.`type` }' with type `${ idx.`type` }'
            |  Known index operations:
            |    Array indexed with Int: a[2]
            |    String indexed with Int: str[0] (Returns a character)
            |    Dict indexed with String: genes["PCSK9"]""".stripMargin)
  }

  def eval(ec: EvalContext): () => Any = ((f.`type`, idx.`type`): @unchecked) match {
    case (t: TArray, TInt) =>
      val localT = t
      val localPos = posn
      AST.evalCompose[IndexedSeq[_], Int](ec, f, idx)((a, i) =>
        try {
          if (i < 0)
            a(a.length + i)
          else
            a(i)
        } catch {
          case e: java.lang.IndexOutOfBoundsException =>
            ParserUtils.error(localPos,
              s"""Tried to access index [$i] on array ${ JsonMethods.compact(localT.toJSON(a)) } of length ${ a.length }
                  |  Hint: All arrays in Hail are zero-indexed (`array[0]' is the first element)
                  |  Hint: For accessing `A'-numbered info fields in split variants, `va.info.field[va.aIndex - 1]' is correct""".stripMargin)
          case e: Throwable => throw e
        })

    case (TDict(_), TString) =>
      AST.evalCompose[Map[_, _], String](ec, f, idx)((d, k) =>
        d.asInstanceOf[Map[String, _]]
          .get(k)
          .orNull
      )

    case (TString, TInt) =>
      AST.evalCompose[String, Int](ec, f, idx)((s, i) => s(i).toString)
  }
}

case class SliceArray(posn: Position, f: AST, idx1: Option[AST], idx2: Option[AST]) extends AST(posn, Array(Some(f), idx1, idx2).flatten) {
  override def typecheckThis(): BaseType = f.`type` match {
    case (t: TArray) =>
      if (idx1.exists(_.`type` != TInt) || idx2.exists(_.`type` != TInt))
        parseError(
          s"""invalid slice expression
              |  Expect (array[start:end],  array[:end], or array[start:]) where start and end are integers
              |  Found [${ idx1.map(_.`type`).getOrElse("") }:${ idx2.map(_.`type`).getOrElse("") }]""".stripMargin)
      else
        t
    case _ => parseError(
      s"""invalid slice expression
          |  Only arrays can be sliced.  Found slice operation on type `${ f.`type` }'""".stripMargin)
  }

  def eval(ec: EvalContext): () => Any = {
    val i1 = idx1.getOrElse(Const(posn, 0, TInt))
    idx2 match {
      case (Some(i2)) =>
        AST.evalCompose[IndexedSeq[_], Int, Int](ec, f, i1, i2)((a, ind1, ind2) => a.slice(ind1, ind2))
      case (None) =>
        AST.evalCompose[IndexedSeq[_], Int](ec, f, i1)((a, ind1) => a.slice(ind1, a.length))
    }
  }
}

case class SymRef(posn: Position, symbol: String) extends AST(posn) {
  def eval(ec: EvalContext): () => Any = {
    val localI = ec.st(symbol)._1
    val localA = ec.a
    if (localI < 0)
      () => 0 // FIXME placeholder
    else
      () => localA(localI)
  }

  override def typecheckThis(ec: EvalContext): BaseType = {
    ec.st.get(symbol) match {
      case Some((_, t)) => t
      case None =>
        parseError(
          s"""symbol `$symbol' not found
              |  Available symbols:
              |    ${ ec.st.map { case (id, (_, t)) => s"${ prettyIdentifier(id) }: $t" }.mkString("\n    ") } """.stripMargin)
    }
  }
}

case class If(pos: Position, cond: AST, thenTree: AST, elseTree: AST)
  extends AST(pos, Array(cond, thenTree, elseTree)) {
  override def typecheckThis(ec: EvalContext): BaseType = {
    (thenTree.`type`, elseTree.`type`) match {
      case (thenType, elseType) if thenType == elseType => thenType
      case (thenType: TNumeric, elseType: TNumeric) => TNumeric.promoteNumeric(Set(thenType, elseType))
      case _ =>
        parseError(s"expected same-type `then' and `else' clause, got `${ thenTree.`type` }' and `${ elseTree.`type` }'")
    }
  }

  def eval(ec: EvalContext): () => Any = {
    val f1 = cond.eval(ec)
    val f2 = thenTree.eval(ec)
    val f3 = elseTree.eval(ec)

    val coerce: Any => Any = `type` match {
      case t: TNumeric =>
        (a: Any) =>
          if (a == null)
            null
          else
            t.conv.to(a)
      case _ => identity
    }

    () => {
      val c = f1()
      if (c != null) {
        coerce(if (c.asInstanceOf[Boolean]) f2() else f3())
      } else
        null
    }
  }
}
