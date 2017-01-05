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
  aggregations: ArrayBuffer[(Int, CPS[Any], Aggregator)]) {

  def setAll(args: Any*) {
    var i = 0
    while (i < args.length) {
      a(i) = args(i)
      i += 1
    }
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
    val af = new ArrayBuffer[(Int, CPS[Any], Aggregator)]()
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
}

case class Positioned[T](x: T) extends Positional

sealed abstract class AST(pos: Position, subexprs: Array[AST] = Array.empty) {
  var `type`: Type = _

  def this(posn: Position, subexpr1: AST) = this(posn, Array(subexpr1))

  def this(posn: Position, subexpr1: AST, subexpr2: AST) = this(posn, Array(subexpr1, subexpr2))

  def eval(ec: EvalContext): () => Any

  def evalAggregator(ec: EvalContext): CPS[Any]

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

  def evalAggregator(ec: EvalContext): CPS[Any] = throw new UnsupportedOperationException

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
            s"""`$t' has neither a field nor a method named `$name'
               |  Hint: sum, min, max, etc. have no parentheses when called on an Array:
               |    counts.sum""".stripMargin)
        case otherwise => fatal(otherwise.message)
      }
  }

  def evalAggregator(ec: EvalContext): CPS[Any] =
    FunctionRegistry.lookupAggregatorTransformation(ec)(lhs.`type`, Seq(), rhs)(lhs, Seq())
      .valueOr { x => fatal(x.message) }
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

  def evalAggregator(ec: EvalContext): CPS[Any] = throw new UnsupportedOperationException
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

  def evalAggregator(ec: EvalContext): CPS[Any] = throw new UnsupportedOperationException
}

case class Lambda(posn: Position, param: String, body: AST) extends AST(posn, body) {
  def typecheck(): Type = parseError("non-function context")

  def eval(ec: EvalContext): () => Any = throw new UnsupportedOperationException

  def evalAggregator(ec: EvalContext): CPS[Any] = throw new UnsupportedOperationException
}

case class Apply(posn: Position, fn: String, args: Array[AST]) extends AST(posn, args) {
  override def typecheckThis(): Type = {
    (fn, args) match {
      case ("str", rhs) =>
        if (rhs.length != 1)
          parseError("str expects 1 argument")
        if (!rhs.head.`type`.isRealizable)
          parseError(s"Argument to str has unrealizable type: ${rhs.head.`type`}")
        TString

      case ("json", rhs) =>
        if (rhs.length != 1)
          parseError("json expects 1 argument")
        if (!rhs.head.`type`.isRealizable)
          parseError(s"Argument to json has unrealizable type: ${rhs.head.`type`}")
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
    case ("str", Array(a)) =>
      val t = a.`type`
      val f = a.eval(ec)
      () => t.str(f())

    case ("json", Array(a)) =>
      val t = a.`type`
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

  def evalAggregator(ec: EvalContext): CPS[Any] = throw new UnsupportedOperationException
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

      // not aggregable: TIterable or TDict
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

  def eval(ec: EvalContext): () => Any = ((lhs.`type`, method, args): @unchecked) match {

    case (it: TContainer, _, Array(Lambda(_, param, body), rest@_*)) =>
      val funType = TFunction(Array(it.elementType), body.`type`)

      FunctionRegistry.lookupMethod(ec)(it, funType +: rest.map(_.`type`), method)(lhs, args)
        .valueOr(x => fatal(x.message))

    case (t, _, _) => FunctionRegistry.lookupMethod(ec)(t, args.map(_.`type`).toSeq, method)(lhs, args)
      .valueOr(x => fatal(x.message))
  }

  def evalAggregator(ec: EvalContext): CPS[Any] = ((lhs.`type`, method, args): @unchecked) match {
    case (it: TContainer, _, Array(Lambda(_, param, body), rest@_*)) =>
      val funType = TFunction(Array(it.elementType), body.`type`)

      FunctionRegistry.lookupAggregatorTransformation(ec)(it, funType +: rest.map(_.`type`), method)(lhs, args)
        .valueOr(x => fatal(x.message))

    case (t, _, _) => FunctionRegistry.lookupAggregatorTransformation(ec)(t, args.map(_.`type`).toSeq, method)(lhs, args)
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

  def evalAggregator(ec: EvalContext): CPS[Any] = throw new UnsupportedOperationException
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

  def evalAggregator(ec: EvalContext): CPS[Any] = {
    val localI = ec.st(symbol)._1
    val localA = ec.a

    if (localI < 0)
      (k: Any => Any) => k(0) // FIXME placeholder
    else
      (k: Any => Any) => k(localA(localI))
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

  def evalAggregator(ec: EvalContext): CPS[Any] = throw new UnsupportedOperationException
}
