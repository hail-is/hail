package org.broadinstitute.hail.expr

import org.apache.spark.sql.Row
import org.apache.spark.util.StatCounter
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.stats._
import org.broadinstitute.hail.utils.{FatalException, Interval}
import org.broadinstitute.hail.variant.{AltAllele, Genotype, Locus, Variant}
import org.json4s.jackson.JsonMethods

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.parsing.input.{Position, Positional}
import scala.language.existentials
import scala.reflect.ClassTag
import org.broadinstitute.hail.utils.EitherIsAMonad._

case class EvalContext(st: SymbolTable,
  a: ArrayBuffer[Any],
  aggregations: ArrayBuffer[(Int, () => Any, Aggregator)]) {

  def setAll(args: Any*) {
    args.zipWithIndex.foreach { case (arg, i) => a(i) = arg }
  }

  def set(index: Int, arg: Any) {
    a(index) = arg
  }
}

object EvalContext {
  def apply(symTab: SymbolTable): EvalContext = {
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
    val af = new ArrayBuffer[(Int, () => Any, Aggregator)]()
    EvalContext(symTab, a, af)
  }

  def apply(args: (String, Type)*): EvalContext = {
    EvalContext(args.zipWithIndex
      .map { case ((name, t), i) => (name, (i, t)) }
      .toMap)
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
  def promoteNumeric(t: TNumeric): Type = t

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
  var `type`: Type = _

  def this(posn: Position, subexpr1: AST) = this(posn, Array(subexpr1))

  def this(posn: Position, subexpr1: AST, subexpr2: AST) = this(posn, Array(subexpr1, subexpr2))

  def eval(ec: EvalContext): () => Any

  def typecheckThis(ec: EvalContext): Type = typecheckThis()

  def typecheckThis(): Type = throw new UnsupportedOperationException

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

case class Const(posn: Position, value: Any, t: Type) extends AST(posn) {
  def eval(c: EvalContext): () => Any = {
    val v = value
    () => v
  }

  override def typecheckThis(): Type = t
}

case class Select(posn: Position, lhs: AST, rhs: String) extends AST(posn, lhs) {
  override def typecheckThis(): Type = {
    (lhs.`type`, rhs) match {

      case (t: TStruct, _) =>
        t.selfField(rhs) match {
          case Some(f) => f.`type`
          case None => parseError(
            s"""`$t' has no field `$rhs'
                |  Available fields: [ ${ t.fields.map(x => prettyIdentifier(x.name)).mkString("\n  ") } ]""".stripMargin)
        }

      case (t, name) => FunctionRegistry.lookupMethodReturnType(t, Seq(), name)
        .valueOr {
          case FunctionRegistry.NotFound(name, typ) =>
            parseError(
              s"""`$t' has no field `$rhs'
                  |  Hint: Don't forget empty-parentheses in a method call, e.g.
                  |    gs.filter(g => g.isCalledHomVar).collect()""".stripMargin)
          case otherwise => parseError(otherwise.message)
        }
    }
  }

  def eval(ec: EvalContext): () => Any = ((lhs.`type`, rhs): @unchecked) match {
    case (t: TStruct, _) =>
      val Some(f) = t.selfField(rhs)
      val i = f.index
      AST.evalCompose[Row](ec, lhs)(_.get(i))

    case (t, name) => FunctionRegistry.lookupMethod(ec)(t, Seq(), name)(lhs, Seq())
      .valueOr {
        case FunctionRegistry.NotFound(name, typ) =>
          fatal(
            s"""`$t' has neither a field nor a method named `$name
                |  Hint: sum, min, max, etc. have no parentheses when called on an Array:
                |    counts.sum""".stripMargin)
        case otherwise => fatal(otherwise.message)
      }
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
    elements.foreach(_.typecheck(ec))
    val types = elements.map(_.`type`)
      .map {
        case t: Type => t
        case bt => parseError(s"invalid struct element found: `$bt'")
      }
    TStruct((names, types, names.indices).zipped.map { case (id, t, i) => Field(id, t, i) })
  }

  def eval(ec: EvalContext): () => Any = {
    val f = elements.map(_.eval(ec))
    () => Annotation.fromSeq(f.map(_ ()))
  }
}

case class Lambda(posn: Position, param: String, body: AST) extends AST(posn, body) {
  def typecheck(): Type = parseError("non-function context")

  def eval(ec: EvalContext): () => Any = throw new UnsupportedOperationException
}

case class Apply(posn: Position, fn: String, args: Array[AST]) extends AST(posn, args) {
  override def typecheckThis(): Type = {
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

    case (_, _) => FunctionRegistry.lookupFun(ec)(fn, args.map(_.`type`).toSeq)(args)
      .valueOr(x => fatal(x.message))
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

        // FIXME rest should be evaluated in empty scope if TAggregable
      case (it: TContainer, _, Array(Lambda(_, param, body), rest@_*)) =>
        rest.foreach(_.typecheck(ec))

        val bodyST =
          it match {
            case tagg: TAggregable => tagg.symTab
            case _ => ec.st
          }

        body.typecheck(ec.copy(st = bodyST + ((param, (-1, it.elementType)))))
        val funType = TFunction(Array(it.elementType), body.`type`)
        `type` = FunctionRegistry.lookupMethodReturnType(it, funType +: rest.map(_.`type`), method)
          .valueOr(x => parseError(x.message))

        // hack
      case (t: TAggregable, "hist", _) =>
        args.foreach(_.typecheck(EvalContext()))
        `type` = FunctionRegistry.lookupMethodReturnType(t, args.map(_.`type`).toSeq, method)
          .valueOr(x => parseError(x.message))

      case _ =>
        super.typecheck(ec)
    }
  }

  override def typecheckThis(): Type = {
    val rhsTypes = args.map(_.`type`)
    (lhs.`type`, method, rhsTypes) match {
      case (t, "orElse", Array(t2)) if t == t2 =>
        t

      case (t, _, _) =>
        FunctionRegistry.lookupMethodReturnType(t, args.map(_.`type`).toSeq, method)
          .valueOr(x => parseError(x.message))
    }
  }

  def eval(ec: EvalContext): () => Any = ((lhs.`type`, method, args): @unchecked) match {

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

    case (it: TContainer, _, Array(Lambda(_, param, body), rest@_*)) =>
      val funType = TFunction(Array(it.elementType), body.`type`)

      FunctionRegistry.lookupMethod(ec)(it, funType +: rest.map(_.`type`), method)(lhs, args)
        .valueOr(x => fatal(x.message))

    case (t, _, _) => FunctionRegistry.lookupMethod(ec)(t, args.map(_.`type`).toSeq, method)(lhs, args)
      .valueOr(x => fatal(x.message))
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
      val lhsT = lhs.`type`
      val rhsT = rhs.`type`
      AST.evalCompose[Any, Any](ec, lhs, rhs) { (left, right) => lhsT.str(left) + rhsT.str(right) }
    case ("~", TBoolean) => AST.evalCompose[String, String](ec, lhs, rhs) { (s, t) =>
      s.r.findFirstIn(t).isDefined
    }

    case ("+" | "*" | "-" | "/", TArray(elementType)) =>
      val f: (Any, Any) => Any = ((operation, elementType): @unchecked) match {
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
        case (TArray(_), TArray(_)) =>
          val localPos = posn
          val localOperation = operation
          AST.evalCompose[IndexedSeq[_], IndexedSeq[_]](ec, lhs, rhs) { case (left, right) =>
            if (left.length != right.length) ParserUtils.error(localPos,
              s"""cannot apply operation `$localOperation' to arrays of unequal length
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

  override def typecheckThis(): Type = (lhs.`type`, operation, rhs.`type`) match {
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
  var operandType: Type = null

  def eval(ec: EvalContext): () => Any = ((operation, operandType): @unchecked) match {
    case ("==", _) => AST.evalCompose[Any, Any](ec, lhs, rhs)(_ == _)
    case ("!=", _) => AST.evalCompose[Any, Any](ec, lhs, rhs)(_ != _)
  }

  override def typecheckThis(): Type = {
    operandType = (lhs.`type`, operation, rhs.`type`) match {
      case (lhsType: TNumeric, "==" | "!=", rhsType: TNumeric) =>
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

case class IndexOp(posn: Position, f: AST, idx: AST) extends AST(posn, Array(f, idx)) {
  override def typecheckThis(): Type = (f.`type`, idx.`type`) match {
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
              s"""Invalid array index: tried to access index [$i] on array `@1' of length ${ a.length }
                  |  Hint: All arrays in Hail are zero-indexed (`array[0]' is the first element)
                  |  Hint: For accessing `A'-numbered info fields in split variants, `va.info.field[va.aIndex - 1]' is correct""".stripMargin,
              JsonMethods.compact(localT.toJSON(a)))
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
  override def typecheckThis(): Type = f.`type` match {
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

    val localSymbol = symbol

    if (localI < 0)
      () => 0 // FIXME placeholder
    else
      () => localA(localI)
  }

  override def typecheckThis(ec: EvalContext): Type = {
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
  override def typecheckThis(ec: EvalContext): Type = {
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

case class Splat(pos: Position, lhs: AST) extends AST(pos, lhs) {
  override def typecheckThis(): Type = {
    lhs.`type` match {
      case t: TStruct => TSplat(t)
      case t => parseError(
        s"""splatting ( <identifier>.* ) operations are only supported on `Struct'
            |  Found `$t'
         """.stripMargin)
    }
  }

  override def eval(ec: EvalContext): () => Any = {
    val nElem = `type`.asInstanceOf[TSplat].struct.size
    val f = lhs.eval(ec)
    () => (f(): @unchecked) match {
      case null => (0 until nElem).map(_ => null)
      case r: Row =>
        assert(r.size == nElem)
        (0 until nElem).map(r.get)
    }
  }
}
