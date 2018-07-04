package is.hail.expr

import is.hail.expr.ir.{AggOp, AggSignature, IR}
import is.hail.expr.ToIRErr._
import is.hail.expr.ir.functions.IRFunctionRegistry
import is.hail.expr.types._
import is.hail.utils.EitherIsAMonad._
import is.hail.utils._
import is.hail.variant.ReferenceGenome

import scala.language.existentials
import scala.reflect.ClassTag
import scala.util.parsing.input.{Position, Positional}

case class EvalContext(st: SymbolTable) {
  st.foreach {
    case (name, (i, t: TAggregable)) =>
      require(t.symTab.exists { case (_, (j, t2)) => j == i && t2 == t.elementType },
        s"did not find binding for type ${ t.elementType } at index $i in agg symbol table for `$name'")
    case _ =>
  }
}

object EvalContext {
  def apply(args: (String, Type)*): EvalContext = {
    EvalContext(args.zipWithIndex
      .map { case ((name, t), i) => (name, (i, t)) }
      .toMap)
  }
}

object DoubleNumericConversion {
  def to(numeric: Any): Double = numeric match {
    case i: Int => i
    case l: Long => l
    case f: Float => f
    case d: Double => d
  }
}

object AST extends Positional {

}

case class Positioned[T](x: T) extends Positional

sealed abstract class AST(pos: Position, val subexprs: Array[AST] = Array.empty) {
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

  def toIROptNoWarning(agg: Option[(String, String)] = None): Option[IR] =
    toIR(agg).toOption

  def toIROpt(agg: Option[(String, String)] = None): Option[IR] = toIR(agg) match {
    case ToIRSuccess(ir) => Some(ir)
    case ToIRFailure(fails) =>
      val context = Thread.currentThread().getStackTrace().drop(2).take(5)
      val caller = context(0)
      val widerContextString = context.map("  " + _).mkString("\n")
      val reasons = fails.map { case (ir, message, context) =>
        s"$message, ${PrettyAST(ir, multiline=false)}, $context"
      }.map("  " + _).mkString("\n")
      log.warn(s"""[${caller} found no AST to IR conversion for:
                  |${ PrettyAST(this, 2) }
                  |due to the following errors:
                  |$reasons
                  |in
                  |$widerContextString""".stripMargin)
      None
  }

  def toIR(agg: Option[(String, String)] = None): ToIRErr[IR]

  def toAggIR(agg: (String, String), cont: (IR) =>  IR): ToIRErr[IR] = ToIRErr.fail(this)
}

case class Const(posn: Position, value: Any, t: Type) extends AST(posn) {
  override def typecheckThis(): Type = t

  def toIR(agg: Option[(String, String)] = None): ToIRErr[IR] = (t, value) match {
    case (t: TInt32, x: Int) => success(ir.I32(x))
    case (t: TInt64, x: Long) => success(ir.I64(x))
    case (t: TFloat32, x: Float) => success(ir.F32(x))
    case (t: TFloat64, x: Double) => success(ir.F64(x))
    case (t: TBoolean, x: Boolean) => success(if (x) ir.True() else ir.False())
    case (t: TString, x: String) => success(ir.Str(x))
    case (t, null) => success(ir.NA(t))
    case _ => throw new RuntimeException(s"Unrecognized constant of type $t: $value")
  }
}

case class SelectAST(posn: Position, lhs: AST, rhs: String) extends AST(posn, lhs) {
  override def typecheckThis(): Type = {
    (lhs.`type`, rhs) match {

      case (t: TStruct, _) =>
        t.selfField(rhs) match {
          case Some(f) => -f.typ
          case None => parseError(
            s"""Struct has no field `$rhs'
               |  Available fields:
               |    ${ t.fields.map(x => s"${ prettyIdentifier(x.name) }: ${ x.typ }").mkString("\n    ") }""".stripMargin)
        }

      case (t, name) => FunctionRegistry.lookupFieldReturnType(t, FastSeq(), name)
        .valueOr {
          case FunctionRegistry.NotFound(name, typ, _) =>
            parseError(s"""`$t' has no field or method `$rhs'""".stripMargin)
          case otherwise => parseError(otherwise.message)
        }
    }
  }

  def toIR(agg: Option[(String, String)] = None): ToIRErr[IR] =
    for {
      base <- lhs.toIR(agg)
      field <- ToIRErr.orElse[IR](
        for {
          t <- whenOfType[TStruct](lhs)
          f <- fromOption(this, s"$t must have field $rhs", t.selfField(rhs))
        } yield ir.GetField(base, rhs),
        for {
          f <- fromOption(this, s"${base.typ} has no method $rhs", IRFunctionRegistry.lookupConversion(rhs, Seq(base.typ)))
        } yield f(Seq(base)))
    } yield field
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

  def toIR(agg: Option[(String, String)] = None): ToIRErr[IR] = for {
    irElements <- all(elements.map(_.toIR(agg)))
    elementTypes = elements.map(_.`type`).distinct
    _ <- blameWhen(this, "array elements must have same type, found: $elementTypes",  elementTypes.length != 1)
  } yield ir.MakeArray(irElements, `type`.asInstanceOf[TArray])
}

abstract class BaseStructConstructor(posn: Position, elements: Array[AST]) extends AST(posn, elements) {
  override def typecheckThis(ec: EvalContext): Type

  def toIR(agg: Option[(String, String)] = None): ToIRErr[IR]
}

case class StructConstructor(posn: Position, names: Array[String], elements: Array[AST]) extends BaseStructConstructor(posn, elements) {
  override def typecheckThis(ec: EvalContext): Type = {
    elements.foreach(_.typecheck(ec))
    val types = elements.map(_.`type`)
    TStruct((names, types, names.indices).zipped.map { case (id, t, i) => Field(id, t, i) })
  }

  def toIR(agg: Option[(String, String)] = None): ToIRErr[IR] = for {
    irElements <- all(elements.map(x => x.toIR(agg).map(ir => (x.`type`, ir))))
    fields = names.zip(irElements).map { case (x, (y, z)) => (x, z) }
  } yield ir.MakeStruct(fields)
}

case class TupleConstructor(posn: Position, elements: Array[AST]) extends BaseStructConstructor(posn, elements) {
  override def typecheckThis(ec: EvalContext): Type = {
    elements.foreach(_.typecheck(ec))
    val types = elements.map(_.`type`)
    TTuple(types)
  }

  def toIR(agg: Option[(String, String)] = None): ToIRErr[IR] = for {
    irElements <- all(elements.map(x => x.toIR(agg)))
  } yield ir.MakeTuple(irElements)
}

case class ReferenceGenomeDependentFunction(posn: Position, fName: String, grName: String, args: Array[AST]) extends AST(posn, args) {
  val rg = ReferenceGenome.getReference(grName)

  val rTyp = fName match {
    case "Locus" => rg.locusType
    case "LocusInterval" => rg.intervalType
    case "LocusAlleles" => TStruct("locus" -> rg.locusType, "alleles" -> TArray(TString()))
    case "getReferenceSequence" => TString()
    case "isValidContig" | "isValidLocus" => TBoolean()
    case "liftoverLocus" | "liftoverLocusInterval" =>
      val destRG = args.head match {
        case Const(_, name, TString(_)) => ReferenceGenome.getReference(name.asInstanceOf[String])
        case _ => fatal(s"invalid arguments to '$fName'.")
      }
      if (fName == "liftoverLocus") destRG.locusType else destRG.intervalType
    case "globalPosToLocus" => rg.locusType
    case "locusToGlobalPos" => TInt64()
    case _ => throw new UnsupportedOperationException
  }

  override def typecheckThis(): Type = {
    fName match {
      case "getReferenceSequence" =>
        (args.map(_.`type`): @unchecked) match {
          case Array(TString(_), TInt32(_), TInt32(_), TInt32(_)) =>
        }
      case "isValidContig" =>
        (args.map(_.`type`): @unchecked) match {
          case Array(TString(_)) =>
        }
      case "isValidLocus" =>
        (args.map(_.`type`): @unchecked) match {
          case Array(TString(_), TInt32(_)) =>
        }
      case "liftoverLocus" =>
        (args.map(_.`type`): @unchecked) match {
          case Array(TString(_), TLocus(rg, _), TFloat64(_)) =>
        }
      case "liftoverLocusInterval" =>
        (args.map(_.`type`): @unchecked) match {
          case Array(TString(_), TInterval(TLocus(rg, _), _), TFloat64(_)) =>
        }
      case "globalPosToLocus" =>
        (args.map(_.`type`): @unchecked) match {
          case Array(TInt64(_)) =>
        }
      case "locusToGlobalPos" =>
        (args.map(_.`type`): @unchecked) match {
          case Array(TLocus(rg, _)) =>
        }
      case _ =>
    }
    rTyp
  }

  def toIR(agg: Option[(String, String)] = None): ToIRErr[IR] = {
    val (frName, actualArgs) = fName match {
      case "liftoverLocus" | "liftoverLocusInterval" =>
        val destRG = ReferenceGenome.getReference(args(0).asInstanceOf[Const].value.asInstanceOf[String])
        (destRG.wrapFunctionName(rg.wrapFunctionName(fName)), args.tail)
      case _ => (rg.wrapFunctionName(fName), args)
    }
    for {
      irArgs <- all(actualArgs.map(_.toIR(agg)))
      ir <- fromOption(
        this,
        s"no RG dependent function found for $frName(${irArgs.map(_.typ).mkString(", ")})",
        IRFunctionRegistry.lookupConversion(frName, irArgs.map(_.typ))
          .map { irf => irf(irArgs) })
    } yield ir
  }
}

case class Lambda(posn: Position, param: String, body: AST) extends AST(posn, body) {
  def typecheck(): Type = parseError("non-function context")

  def toIR(agg: Option[(String, String)] = None): ToIRErr[IR] = fail(this)
}

case class ApplyAST(posn: Position, fn: String, args: Array[AST]) extends AST(posn, args) {
  assert(fn != "LocusInterval")

  override def typecheckThis(): Type = {
    (fn, args) match {
      case ("annotate", rhs) =>
        val (t1, t2) = args.map(_.`type`) match {
          case Array(t1: TStruct, t2: TStruct) => (t1, t2)
          case other => parseError(
            s"""invalid arguments to `$fn'
               |  Expected $fn(Struct, Struct), found $fn(${ other.mkString(", ") })""".stripMargin)
        }

        t1.annotate(t2)._1

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
    def needsSymRef(other: AST) = parseError(
      s"""invalid arguments for method `$fn'
         |  Expected struct field identifiers as arguments, but found a `${ other.getClass.getSimpleName }' expression
         |  Usage: $fn(Array[Struct], key identifier)""".stripMargin)
    fn match {
      case "select" =>
        val struct = args(0)
        struct.typecheck(ec)
        val identifiers = args.tail.map {
          case SymRefAST(_, id) => id
          case badAST => needsSymRef(badAST)
        }
        assert(identifiers.duplicates().isEmpty)
        `type` = struct.`type`.asInstanceOf[TStruct].select(identifiers)._1

      case "drop" =>
        val struct = args(0)
        struct.typecheck(ec)
        val identifiers = args.tail.map {
          case SymRefAST(_, id) => id
          case badAST => needsSymRef(badAST)
        }
        assert(identifiers.duplicates().isEmpty)
        `type` = struct.`type`.asInstanceOf[TStruct].filterSet(identifiers.toSet, include = false)._1

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

  private def tryPrimOpConversion: Seq[IR] => Option[IR] = flatLift {
      case Seq(x) => for {
        op <- ir.UnaryOp.fromString.lift(fn)
        t <- ir.UnaryOp.returnTypeOption(op, x.typ)
      } yield ir.ApplyUnaryPrimOp(op, x)
      case Seq(x, y) => for {
        op <- ir.BinaryOp.fromString.lift(fn)
        t <- ir.BinaryOp.returnTypeOption(op, x.typ, y.typ)
      } yield ir.ApplyBinaryPrimOp(op, x, y)
    }

  private def tryComparisonConversion: Seq[IR] => Option[IR] = flatLift {
    case Seq(x, y) => for {
      op <- ir.ComparisonOp.fromStringAndTypes.lift((fn, x.typ, y.typ))
    } yield ir.ApplyComparisonOp(op, x, y)
  }

  private[this] def tryIRConversion(agg: Option[(String, String)]): ToIRErr[IR] =
    for {
      irArgs <- all(args.map(_.toIR(agg)))
      ir <- ToIRErr.exactlyOne[IR](
        this,
        tryPrimOpConversion(irArgs),
        tryComparisonConversion(irArgs),
        IRFunctionRegistry
          .lookupConversion(fn, irArgs.map(_.typ))
          .map { irf => irf(irArgs) })
    } yield ir

  def toIR(agg: Option[(String, String)] = None): ToIRErr[IR] = {
    fn match {
      case "select" =>
        for (structIR <- args(0).toIR(agg)) yield {
          val identifiers = args.tail.map { case SymRefAST(_, id) => id }
          ir.SelectFields(structIR, identifiers)
        }
      case "drop" =>
        for (structIR <- args(0).toIR(agg)) yield {
          val t = types.coerce[TStruct](structIR.typ)
          val identifiers = args.tail.map { case SymRefAST(_, id) => id }.toSet
          val keep = t.fieldNames.filter(!identifiers.contains(_))
          ir.SelectFields(structIR, keep)
        }
      case "uniroot" =>
        val IndexedSeq(Lambda(_, param, body), min, max) = args.toFastIndexedSeq
        for {
          irF <- body.toIR(agg)
          minIR <- min.toIR(agg)
          maxIR <- max.toIR(agg)
        } yield ir.Uniroot(param, irF, minIR, maxIR)
      case _ =>
        tryIRConversion(agg)
    }
  }
}

case class ApplyMethodAST(posn: Position, lhs: AST, method: String, args: Array[AST]) extends AST(posn, lhs +: args) {
  def getSymRefId(ast: AST): String = {
    ast match {
      case SymRefAST(_, id) => id
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

      case (tt: TTuple, "[]", Array(idxAST@Const(_, v, t))) =>
        idxAST.typecheck(ec)
        val idx = v.asInstanceOf[Int]
        assert(idx >= 0 && idx < tt.size)
        `type` = tt.types(idx)

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

  override def toAggIR(agg: (String, String), cont: (IR) => IR): ToIRErr[IR] = {
    assert(lhs.`type`.isInstanceOf[TAggregable])
    assert(args.length == 1)
    val Lambda(_, name, body) = args(0)
    method match {
      case "map" =>
        for {
          bodyx <- body.toIR()
          rx <- lhs.toAggIR(agg, lhsx =>
            cont(ir.Let(name, lhsx, bodyx)))
        } yield rx
      case "filter" =>
        for {
          bodyx <- body.toIR()
          rx <- lhs.toAggIR(agg, { lhsx =>
            val lhsv = ir.genUID()
            ir.Let(lhsv, lhsx,
              ir.If(ir.Let(name, ir.Ref(lhsv, lhsx.typ), bodyx),
                cont(ir.Ref(lhsv, lhsx.typ)),
                ir.Begin(FastIndexedSeq.empty[IR])))
          })
        } yield rx
      case "flatMap" =>
        for {
          bodyx <- body.toIR()
          rx <- lhs.toAggIR(agg, { lhsx =>
            val t = -lhsx.typ.asInstanceOf[TContainer].elementType
            val v = ir.genUID()
            ir.ArrayFor(ir.ToArray(ir.Let(name, lhsx, bodyx)),
              v,
              cont(ir.Ref(v, t)))
          })
        } yield rx
    }
  }

  def toIR(agg: Option[(String, String)] = None): ToIRErr[IR] = {
    (lhs.`type`, method, args: IndexedSeq[AST]) match {
      case (t: TAggregable, "callStats", IndexedSeq(Lambda(_, name, body))) =>
        for {
          op <- fromOption(
            this,
            s"no AggOp for method $method",
            AggOp.fromString.lift(method))
          bodyx <- body.toIR()
          initOpArgs = Some(FastIndexedSeq(bodyx))
          aggSig = AggSignature(op,
            FastIndexedSeq(),
            initOpArgs.map(_.map(_.typ)),
            FastIndexedSeq(-t.elementType))
          ca <- fromOption(
            this,
            "no CodeAggregator",
            AggOp.getOption(aggSig))
          rx <- lhs.toAggIR(agg.get, x => ir.SeqOp(ir.I32(0), FastIndexedSeq(x), aggSig))
        } yield
          ir.ApplyAggOp(rx, FastIndexedSeq(), initOpArgs, aggSig): IR
      case (t: TAggregable, "inbreeding", IndexedSeq(Lambda(_, name, body))) =>
        for {
          op <- fromOption(
            this,
            s"no AggOp for method $method",
            AggOp.fromString.lift(method))
          bodyx <- body.toIR()
          aggSig = AggSignature(op,
            FastIndexedSeq(),
            None,
            FastIndexedSeq(-t.elementType, bodyx.typ))
          ca <- fromOption(
            this,
            "no CodeAggregator",
            AggOp.getOption(aggSig))
          rx <- lhs.toAggIR(agg.get, x => ir.SeqOp(ir.I32(0), FastIndexedSeq(x, ir.Let(name, x, bodyx)), aggSig))
        } yield
          ir.ApplyAggOp(rx, FastIndexedSeq(), None, aggSig): IR
      case (t: TAggregable, "takeBy", IndexedSeq(Lambda(_, name, body), n)) =>
        for {
          op <- fromOption(
            this,
            s"no AggOp for method $method",
            AggOp.fromString.lift(method))
          nx <- n.toIR()
          bodyx <- body.toIR()
          aggSig = AggSignature(op,
            FastIndexedSeq(nx.typ),
            None,
            FastIndexedSeq(-t.elementType, bodyx.typ))
          ca <- fromOption(
            this,
            "no CodeAggregator",
            AggOp.getOption(aggSig))
          rx <- lhs.toAggIR(agg.get, x => ir.SeqOp(ir.I32(0), FastIndexedSeq(x, ir.Let(name, x, bodyx)), aggSig))
        } yield
          ir.ApplyAggOp(rx, FastIndexedSeq(nx), None, aggSig): IR
      case (t: TAggregable, "fraction", IndexedSeq(Lambda(_, name, body))) =>
        for {
          op <- fromOption(
            this,
            s"no AggOp for method $method",
            AggOp.fromString.lift(method))
          bodyx <- body.toIR()
          aggSig = AggSignature(op,
            FastIndexedSeq(),
            None,
            FastIndexedSeq(bodyx.typ))
          ca <- fromOption(
            this,
            "no CodeAggregator",
            AggOp.getOption(aggSig))
          rx <- lhs.toAggIR(agg.get, x => ir.SeqOp(ir.I32(0), FastIndexedSeq(ir.Let(name, x, bodyx)), aggSig))
        } yield
          ir.ApplyAggOp(rx, FastIndexedSeq(), None, aggSig): IR
      case (t: TAggregable, "count", IndexedSeq()) =>
        for {
          op <- fromOption(
            this,
            s"no AggOp for method $method",
            AggOp.fromString.lift(method))
          aggSig = AggSignature(op,
            FastIndexedSeq(),
            None,
            FastIndexedSeq())
          ca <- fromOption(
            this,
            "no CodeAggregator",
            AggOp.getOption(aggSig))
          rx <- lhs.toAggIR(agg.get, x => ir.SeqOp(ir.I32(0), FastIndexedSeq(), aggSig))
        } yield
          ir.ApplyAggOp(rx, FastIndexedSeq(), None, aggSig): IR
      case (t: TAggregable, _, _) =>
        for {
          op <- fromOption(
            this,
            s"no AggOp for method $method",
            AggOp.fromString.lift(method))
          constructorArgs <- all(args.map(_.toIR(agg))).map(_.toFastIndexedSeq)
          aggSig = AggSignature(op,
            constructorArgs.map(_.typ),
            None,
            FastIndexedSeq(-t.elementType))
          ca <- fromOption(
            this,
            "no CodeAggregator",
            AggOp.getOption(aggSig))
          rx <- lhs.toAggIR(agg.get, x => ir.SeqOp(
            ir.I32(0),
            FastIndexedSeq(x),
            aggSig))
        } yield
          ir.ApplyAggOp(rx, constructorArgs, None, aggSig): IR
      case (t, m, IndexedSeq(Lambda(_, name, body))) =>
        for {
          a <- lhs.toIR(agg)
          b <- body.toIR(agg)
          result <- fromOption(
            this,
            s"no method $m on type $t",
            optMatch((t, m)) {
              case (_: TArray, "map") => ir.ArrayMap(a, name, b)
              case (_: TArray, "filter") => ir.ArrayFilter(a, name, b)
              case (_: TArray, "flatMap") => ir.ArrayFlatMap(a, name, b)
              case (_: TSet, "flatMap") =>
                ir.ToSet(
                  ir.ArrayFlatMap(ir.ToArray(a), name,
                      ir.ToArray(b)))
              case (_: TArray, "exists") =>
                val v = ir.genUID()
                ir.ArrayFold(a, ir.False(), v, name, ir.ApplySpecial("||", FastSeq(ir.Ref(v, TBoolean()), b)))
              case (_: TArray, "forall") =>
                val v = ir.genUID()
                ir.ArrayFold(a, ir.True(), v, name, ir.ApplySpecial("&&", FastSeq(ir.Ref(v, TBoolean()), b)))
              case (_: TSet, "map") => ir.ToSet(ir.ArrayMap(ir.ToArray(a), name, b))
              case (_: TSet, "filter") => ir.ToSet(ir.ArrayFilter(ir.ToArray(a), name, b))
              case (_: TSet, "exists") =>
                val v = ir.genUID()
                ir.ArrayFold(ir.ToArray(a), ir.False(), v, name, ir.ApplySpecial("||", FastSeq(ir.Ref(v, TBoolean()), b)))
              case (_: TSet, "forall") =>
                val v = ir.genUID()
                ir.ArrayFold(ir.ToArray(a), ir.True(), v, name, ir.ApplySpecial("&&", FastSeq(ir.Ref(v, TBoolean()), b)))
              case (_: TDict, "mapValues") =>
                val newName = ir.Ref(name, -types.coerce[TContainer](a.typ).elementType)
                val value = ir.Subst(b, ir.Env.empty.bind(name -> ir.GetField(newName, "value")))
                ir.ToDict(
                  ir.ArrayMap(
                    ir.ToArray(a),
                    name,
                    ir.MakeStruct(Seq(
                      "key" -> ir.GetField(newName, "key"),
                      "value" -> value))))

              case (_: TArray, "groupBy") =>
                ir.GroupByKey(ir.ArrayMap(a, name, ir.MakeTuple(FastSeq(b, ir.Ref(name, types.coerce[TContainer](a.typ).elementType)))))
              case (_: TSet, "groupBy") =>
                ir.GroupByKey(ir.ArrayMap(ir.ToArray(a), name, ir.MakeTuple(FastSeq(b, ir.Ref(name, types.coerce[TContainer](a.typ).elementType)))))
              case (_: TArray, "sortBy") =>
                val ref = ir.Ref(ir.genUID(), TTuple(b.typ, types.coerce[TContainer](a.typ).elementType))
                ir.ArrayMap(
                  ir.ArraySort(
                    ir.ArrayMap(a, name,
                      ir.MakeTuple(FastSeq(b, ir.Ref(name, types.coerce[TContainer](a.typ).elementType)))),
                    true, onKey = true),
                  ref.name,
                  ir.GetTupleElement(ref, 1))
            })
        } yield result
      case (t, m, IndexedSeq(Lambda(_, name, body), arg1)) =>
        for {
          a <- lhs.toIR(agg)
          b <- body.toIR(agg)
          irArg1 <- arg1.toIR(agg)
          result <- fromOption(
            this,
            s"no method $m on type $t",
            optMatch((t, m)) {
              case (_: TArray, "sortBy") =>
                val ref = ir.Ref(ir.genUID(), TTuple(b.typ, types.coerce[TContainer](a.typ).elementType))
                ir.ArrayMap(
                  ir.ArraySort(
                    ir.ArrayMap(a, name,
                      ir.MakeTuple(FastSeq(b, ir.Ref(name, types.coerce[TContainer](a.typ).elementType)))),
                    irArg1, onKey = true),
                  ref.name,
                  ir.GetTupleElement(ref, 1))
            }
          )
        } yield result
      case _ =>
        for {
          irs <- all((lhs +: args).map(_.toIR(agg)))
          argTypes = irs.map(_.typ)
          ir <- fromOption(
            this,
            s"no method $method on type ${lhs.`type`} with arguments ${argTypes.tail}",
            IRFunctionRegistry.lookupConversion(method, argTypes)
              .map { irf => irf(irs) })
        } yield ir
    }
  }
}

case class LetAST(posn: Position, bindings: Array[(String, AST)], body: AST) extends AST(posn, bindings.map(_._2) :+ body) {

  override def typecheck(ec: EvalContext) {
    var symTab2 = ec.st
    for ((id, v) <- bindings) {
      v.typecheck(ec.copy(st = symTab2))
      symTab2 = symTab2 + (id -> (-1, v.`type`))
    }
    body.typecheck(ec.copy(st = symTab2))

    `type` = body.`type`
  }

  def toIR(agg: Option[(String, String)] = None): ToIRErr[IR] = for {
    irBindings <- all(bindings.map { case (x, y) => y.toIR(agg).map(irY => (x, irY)) })
    irBody <- body.toIR(agg)
  } yield irBindings.foldRight(irBody) { case ((name, v), x) => ir.Let(name, v, x) }
}

case class SymRefAST(posn: Position, symbol: String) extends AST(posn) {
  override def typecheckThis(ec: EvalContext): Type = {
    ec.st.get(symbol) match {
      case Some((_, t)) =>
        -t
      case None =>
        val symbols = ec.st.toArray.sortBy(_._2._1).map { case (id, (_, t)) => s"${ prettyIdentifier(id) }: $t" }
        parseError(
          s"""symbol `$symbol' not found
             |  Available symbols:
             |    ${ symbols.mkString("\n    ") }""".stripMargin)
    }
  }

  override def toAggIR(agg: (String, String), cont: (IR) => IR): ToIRErr[IR] = {
    assert(symbol == agg._1)
    success(cont(ir.Ref(agg._2, -`type`.asInstanceOf[TAggregable].elementType)))
  }

  def toIR(agg: Option[(String, String)] = None): ToIRErr[IR] =
    success(ir.Ref(symbol, `type`))
}

case class IfAST(pos: Position, cond: AST, thenTree: AST, elseTree: AST)
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

  def toIR(agg: Option[(String, String)] = None): ToIRErr[IR] = for {
    condition <- cond.toIR(agg)
    consequent <- thenTree.toIR(agg)
    alternate <- elseTree.toIR(agg)
  } yield ir.If(condition, consequent, alternate)
}

// PrettyAST(ast) gives a pretty-print of an AST tree

object PrettyAST {
  def apply(rootAST: AST, depth: Int = 0, multiline: Boolean = true): String = {
    val sb = new StringBuilder()

    def putDeepAST(ast: AST, depth: Int) {
      if (multiline)
        sb.append("  " * depth)
      val children = ast.subexprs
      sb.append(astToName(ast))
      if (children.length > 0) {
        sb.append("(")
        if (multiline)
          sb.append("\n")
        var i = 0
        while (i < children.length) {
          putDeepAST(children(i), depth+1)

          if (multiline) sb.append("\n")
          else if (i + 1 != children.length)
            sb.append(" ")
          i += 1
        }
        if (multiline)
          sb.append("  " * depth)
        sb.append(")")
      }
    }

    putDeepAST(rootAST, depth)
    sb.toString()
  }

  private def astToName(ast: AST): String = {
    ast match {
      case ApplyAST(_, fn, _) => s"Apply[${fn}]"
      case ApplyMethodAST(_, _, method, _) => s"ApplyMethod[${method}]"
      case ArrayConstructor(_, _) => "ArrayConstructor"
      case Const(_, value, _) => s"Const[${Option(value).getOrElse("NA")}]"
      case IfAST(_, _, _, _) => "If"
      case Lambda(_, param, _) => s"Lambda[${param}]"
      case LetAST(_, _, _) => "Let"
      case SelectAST(_, _, rhs) => s"Select[${rhs}]"
      case StructConstructor(_, _, _) => "StructConstructor"
      case SymRefAST(_, symbol) => s"SymRef[${symbol}]"
      case TupleConstructor(_, _) => "TupleConstructor"
      case _ => "UnknownAST"
    }
  }
}

