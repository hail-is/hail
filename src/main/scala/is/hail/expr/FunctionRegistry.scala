package is.hail.expr

import breeze.linalg.DenseVector
import is.hail.annotations.Annotation
import is.hail.stats._
import is.hail.utils._
import is.hail.variant.{AltAllele, Genotype, Locus, Variant}
import is.hail.methods._

import scala.collection.mutable
import is.hail.utils.EitherIsAMonad._
import org.json4s.jackson.JsonMethods

import scala.language.higherKinds

case class FunctionMetadata(docstring: Option[String] = None, category: Option[String] = None,
  argNames: Array[String] = Array.empty[String])

object FunctionRegistry {

  sealed trait LookupError {
    def message: String
  }

  sealed case class NotFound(name: String, typ: TypeTag) extends LookupError {
    def message = s"No function found with name `$name' and argument ${ plural(typ.xs.size, "type") } $typ"
  }

  sealed case class Ambiguous(name: String, typ: TypeTag, alternates: Seq[(Int, (TypeTag, Fun))]) extends LookupError {
    def message =
      s"""found ${ alternates.size } ambiguous matches for $typ:
          |  ${ alternates.map(_._2._1).mkString("\n  ") }""".stripMargin
  }

  type Err[T] = Either[LookupError, T]

  private val registry = mutable.HashMap[String, Seq[(TypeTag, Fun, FunctionMetadata)]]().withDefaultValue(Seq.empty)

  private val conversions = new mutable.HashMap[(Type, Type), (Int, UnaryFun[Any, Any])]

  private def lookupConversion(from: Type, to: Type): Option[(Int, UnaryFun[Any, Any])] = conversions.get(from -> to)

  private def registerConversion[T, U](how: T => U, priority: Int = 1)(implicit hrt: HailRep[T], hru: HailRep[U]) {
    val from = hrt.typ
    val to = hru.typ
    require(priority >= 1)
    lookupConversion(from, to) match {
      case Some(_) =>
        throw new RuntimeException(s"The conversion between $from and $to is already bound")
      case None =>
        conversions.put(from -> to, priority -> UnaryFun[Any, Any](to, x => how(x.asInstanceOf[T])))
    }
  }

  private def lookup(name: String, typ: TypeTag): Err[Fun] = {

    val matches = registry(name).flatMap { case (tt, f, _) =>
      tt.clear()
      if (tt.xs.size == typ.xs.size) {
        val conversions = (tt.xs, typ.xs).zipped.map { case (l, r) =>
          if (l.isBound) {
            if (l.unify(r))
              Some(None)
            else {
              val conv = lookupConversion(r, l).map(c => Some(c))
              conv
            }
          } else if (l.unify(r)) {
            Some(None)
          } else
            None
        }

        anyFailAllFail[Array, Option[(Int, UnaryFun[Any, Any])]](conversions)
          .map { arr =>
            if (arr.forall(_.isEmpty))
              0 -> (tt.subst(), f.subst())
            else {
              val arr2 = arr.map(_.getOrElse(0 -> UnaryFun[Any, Any](null, (a: Any) => a)))
              arr2.map(_._1).max -> (tt.subst(), f.subst().convertArgs(arr2.map(_._2)))
            }
          }
      } else
        None
    }.groupBy(_._1).toArray.sortBy(_._1)

    matches.headOption
      .toRight[LookupError](NotFound(name, typ))
      .flatMap { case (priority, it) =>
        assert(it.nonEmpty)
        if (it.size == 1)
          Right(it.head._2._2)
        else {
          assert(priority != 0)
          Left(Ambiguous(name, typ, it))
        }
      }
  }

  private def bind(name: String, typ: TypeTag, f: Fun, md: FunctionMetadata) = {
    registry.updateValue(name, Seq.empty, (typ, f, md) +: _)
  }

  def lookupMethodReturnType(typ: Type, typs: Seq[Type], name: String): Err[Type] =
    lookup(name, MethodType(typ +: typs: _*)).map(_.retType)

  def lookupMethod(ec: EvalContext)(typ: Type, typs: Seq[Type], name: String)(lhs: AST, args: Seq[AST]): Err[() => Any] = {
    require(typs.length == args.length)

    val m = lookup(name, MethodType(typ +: typs: _*))
    m.map {
      case aggregator: Arity0Aggregator[_, _] =>
        val localA = ec.a
        val idx = localA.length
        localA += null
        ec.aggregations += ((idx, lhs.evalAggregator(ec), aggregator.ctor()))
        () => localA(idx)

      case aggregator: Arity1Aggregator[_, u, _] =>
        val localA = ec.a
        val idx = localA.length
        localA += null

        val u = args(0).eval(EvalContext())()

        if (u == null)
          fatal("Argument evaluated to missing in call to aggregator $name")

        ec.aggregations += ((idx, lhs.evalAggregator(ec), aggregator.ctor(
          u.asInstanceOf[u])))
        () => localA(idx)

      case aggregator: Arity3Aggregator[_, u, v, w, _] =>
        val localA = ec.a
        val idx = localA.length
        localA += null

        val u = args(0).eval(EvalContext())()
        val v = args(1).eval(EvalContext())()
        val w = args(2).eval(EvalContext())()

        if (u == null)
          fatal("Argument 1 evaluated to missing in call to aggregator $name")
        if (v == null)
          fatal("Argument 2 evaluated to missing in call to aggregator $name")
        if (w == null)
          fatal("Argument 3 evaluated to missing in call to aggregator $name")

        ec.aggregations += ((idx, lhs.evalAggregator(ec), aggregator.ctor(
          u.asInstanceOf[u],
          v.asInstanceOf[v],
          w.asInstanceOf[w])))
        () => localA(idx)

      case aggregator: UnaryLambdaAggregator[t, u, v] =>
        val Lambda(_, param, body) = args(0)

        val idx = ec.a.length
        val localA = ec.a
        localA += null

        val bodyST =
          lhs.`type` match {
            case tagg: TAggregable => tagg.symTab
            case _ => ec.st
          }

        val bodyFn = body.eval(ec.copy(st = bodyST + (param -> (idx, lhs.`type`.asInstanceOf[TContainer].elementType))))
        val g = (x: Any) => {
          localA(idx) = x
          bodyFn()
        }

        ec.aggregations += ((idx, lhs.evalAggregator(ec), aggregator.ctor(g)))
        () => localA(idx)

      case aggregator: BinaryLambdaAggregator[t, u, v, w] =>
        val Lambda(_, param, body) = args(0)

        val idx = ec.a.length
        val localA = ec.a
        localA += null

        val bodyST =
          lhs.`type` match {
            case tagg: TAggregable => tagg.symTab
            case _ => ec.st
          }

        val bodyFn = body.eval(ec.copy(st = bodyST + (param -> (idx, lhs.`type`.asInstanceOf[TContainer].elementType))))
        val g = (x: Any) => {
          localA(idx) = x
          bodyFn()
        }

        val v = args(1).eval(EvalContext())()
        if (v == null)
          fatal("Argument evaluated to missing in call to aggregator $name")

        ec.aggregations += ((idx, lhs.evalAggregator(ec), aggregator.ctor(g, v.asInstanceOf[v])))
        () => localA(idx)

      case f: UnaryFun[_, _] =>
        AST.evalCompose(ec, lhs)(f)
      case f: UnarySpecial[_, _] =>
        val t = lhs.eval(ec)
        () => f(t)
      case f: OptionUnaryFun[_, _] =>
        AST.evalFlatCompose(ec, lhs)(f)
      case f: BinaryFun[_, _, _] =>
        AST.evalCompose(ec, lhs, args(0))(f)
      case f: BinarySpecial[_, _, _] =>
        val t = lhs.eval(ec)
        val u = args(0).eval(ec)
        () => f(t, u)
      case f: BinaryLambdaFun[t, _, _] =>
        val Lambda(_, param, body) = args(0)

        val localIdx = ec.a.length
        val localA = ec.a
        localA += null

        val bodyST =
          lhs.`type` match {
            case tagg: TAggregable => tagg.symTab
            case _ => ec.st
          }

        val bodyFn = body.eval(ec.copy(st = bodyST + (param -> (localIdx, lhs.`type`.asInstanceOf[TContainer].elementType))))
        val g = (x: Any) => {
          localA(localIdx) = x
          bodyFn()
        }

        AST.evalCompose[t](ec, lhs) { x1 => f(x1, g) }
      case f: Arity3LambdaFun[t, _, v, _] =>
        val Lambda(_, param, body) = args(0)

        val localIdx = ec.a.length
        val localA = ec.a
        localA += null

        val bodyST =
          lhs.`type` match {
            case tagg: TAggregable => tagg.symTab
            case _ => ec.st
          }

        val bodyFn = body.eval(ec.copy(st = bodyST + (param -> (localIdx, lhs.`type`.asInstanceOf[TContainer].elementType))))
        val g = (x: Any) => {
          localA(localIdx) = x
          bodyFn()
        }

        AST.evalCompose[t, v](ec, lhs, args(1)) { (x1, x2) => f(x1, g, x2) }
      case f: BinaryLambdaAggregatorTransformer[t, _, _] =>
        throw new RuntimeException(s"Internal hail error, aggregator transformation ($name : $typ, $typs) in non-aggregator position")
      case f: Arity3Fun[_, _, _, _] =>
        AST.evalCompose(ec, lhs, args(0), args(1))(f)
      case f: Arity4Fun[_, _, _, _, _] =>
        AST.evalCompose(ec, lhs, args(0), args(1), args(2))(f)
      case fn =>
        throw new RuntimeException(s"Internal hail error, bad binding in function registry for `$name' with argument types $typ, $typs: $fn")
    }
  }

  def lookupAggregatorTransformation(ec: EvalContext)(typ: Type, typs: Seq[Type], name: String)(lhs: AST, args: Seq[AST]): Err[CPS[Any]] = {
    require(typs.length == args.length)

    lookup(name, MethodType(typ +: typs: _*)).map {
      case f: BinaryLambdaAggregatorTransformer[t, _, _] =>
        val Lambda(_, param, body) = args(0)

        val idx = ec.a.length
        val localA = ec.a
        localA += null

        val bodyST =
          lhs.`type` match {
            case tagg: TAggregable => tagg.symTab
            case _ => ec.st
          }

        val bodyFn = body.eval(ec.copy(st = bodyST + (param -> (idx, lhs.`type`.asInstanceOf[TContainer].elementType))))
        val g = (x: Any) => {
          localA(idx) = x
          bodyFn()
        }

        val t = lhs.evalAggregator(ec)
        f(t, g)
      case _ =>
        throw new RuntimeException(s"Internal hail error, non-aggregator transformation, `$name' with argument types $typ, $typs, found in aggregator position")
    }
  }

  def lookupFun(ec: EvalContext)(name: String, typs: Seq[Type])(args: Seq[AST]): Err[() => Any] = {
    require(typs.length == args.length)

    lookup(name, FunType(typs: _*)).map {
      case f: UnaryFun[_, _] =>
        AST.evalCompose(ec, args(0))(f)
      case f: UnarySpecial[_, _] =>
        val t = args(0).eval(ec)
        () => f(t)
      case f: OptionUnaryFun[_, _] =>
        AST.evalFlatCompose(ec, args(0))(f)
      case f: BinaryFun[_, _, _] =>
        AST.evalCompose(ec, args(0), args(1))(f)
      case f: BinarySpecial[_, _, _] =>
        val t = args(0).eval(ec)
        val u = args(1).eval(ec)
        () => f(t, u)
      case f: Arity3Fun[_, _, _, _] =>
        AST.evalCompose(ec, args(0), args(1), args(2))(f)
      case f: Arity4Fun[_, _, _, _, _] =>
        AST.evalCompose(ec, args(0), args(1), args(2), args(3))(f)
      case fn =>
        throw new RuntimeException(s"Internal hail error, bad binding in function registry for `$name' with argument types $typs: $fn")
    }
  }

  def lookupFunReturnType(name: String, typs: Seq[Type]): Err[Type] =
    lookup(name, FunType(typs: _*)).map(_.retType)

  def registerMethod[T, U](name: String, impl: T => U, md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, MethodType(hrt.typ), UnaryFun[T, U](hru.typ, impl), md)
  }

  def registerMethod[T, U, V](name: String, impl: (T, U) => V, md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, MethodType(hrt.typ, hru.typ), BinaryFun[T, U, V](hrv.typ, impl), md)
  }

  def registerMethodSpecial[T, U, V](name: String, impl: (() => Any, () => Any) => V, md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, MethodType(hrt.typ, hru.typ), BinarySpecial[T, U, V](hrv.typ, impl), md)
  }

  def registerLambdaMethod[T, U, V](name: String, impl: (T, (Any) => Any) => V, md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    val m = BinaryLambdaFun[T, U, V](hrv.typ, impl)
    bind(name, MethodType(hrt.typ, hru.typ), m, md)
  }

  def registerLambdaMethod[T, U, V, W](name: String, impl: (T, (Any) => Any, V) => W, md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    val m = Arity3LambdaFun[T, U, V, W](hrw.typ, impl)
    bind(name, MethodType(hrt.typ, hru.typ, hrv.typ), m, md)
  }

  def registerLambdaAggregatorTransformer[T, U, V](name: String, impl: (CPS[Any], (Any) => Any) => CPS[V], md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    val m = BinaryLambdaAggregatorTransformer[T, U, V](hrv.typ, impl)
    bind(name, MethodType(hrt.typ, hru.typ), m, md)
  }

  def registerMethod[T, U, V, W](name: String, impl: (T, U, V) => W, md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    bind(name, MethodType(hrt.typ, hru.typ, hrv.typ), Arity3Fun[T, U, V, W](hrw.typ, impl), md)
  }

  def register[T, U](name: String, impl: T => U, md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, FunType(hrt.typ), UnaryFun[T, U](hru.typ, impl), md)
  }

  def registerSpecial[T, U](name: String, impl: (() => Any) => U, md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, FunType(hrt.typ), UnarySpecial[T, U](hru.typ, impl), md)
  }

  def registerOptionMethod[T, U](name: String, impl: T => Option[U], md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, MethodType(hrt.typ), OptionUnaryFun[T, U](hru.typ, impl), md)
  }

  def registerOption[T, U](name: String, impl: T => Option[U], md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, FunType(hrt.typ), OptionUnaryFun[T, U](hru.typ, impl), md)
  }

  def registerUnaryNAFilteredCollectionMethod[T, U](name: String, impl: TraversableOnce[T] => U, md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, MethodType(TArray(hrt.typ)), UnaryFun[IndexedSeq[_], U](hru.typ, { (ts: IndexedSeq[_]) =>
      impl(ts.filter(t => t != null).map(_.asInstanceOf[T]))
    }), md)
    bind(name, MethodType(TSet(hrt.typ)), UnaryFun[Set[_], U](hru.typ, { (ts: Set[_]) =>
      impl(ts.filter(t => t != null).map(_.asInstanceOf[T]))
    }), md)
  }

  def register[T, U, V](name: String, impl: (T, U) => V, md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, FunType(hrt.typ, hru.typ), BinaryFun[T, U, V](hrv.typ, impl), md)
  }

  def registerSpecial[T, U, V](name: String, impl: (() => Any, () => Any) => V, md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, FunType(hrt.typ, hru.typ), BinarySpecial[T, U, V](hrv.typ, impl), md)
  }

  def register[T, U, V, W](name: String, impl: (T, U, V) => W, md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    bind(name, FunType(hrt.typ, hru.typ, hrv.typ), Arity3Fun[T, U, V, W](hrw.typ, impl), md)
  }

  def register[T, U, V, W, X](name: String, impl: (T, U, V, W) => X, md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W], hrx: HailRep[X]) = {
    bind(name, FunType(hrt.typ, hru.typ, hrv.typ, hrw.typ), Arity4Fun[T, U, V, W, X](hrx.typ, impl), md)
  }

  def registerAnn[T](name: String, t: TStruct, impl: T => Annotation, md: FunctionMetadata)
    (implicit hrt: HailRep[T]) = {
    register(name, impl, md)(hrt, new HailRep[Annotation] {
      def typ = t
    })
  }

  def registerAnn[T, U](name: String, t: TStruct, impl: (T, U) => Annotation, md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    register(name, impl, md)(hrt, hru, new HailRep[Annotation] {
      def typ = t
    })
  }

  def registerAnn[T, U, V](name: String, t: TStruct, impl: (T, U, V) => Annotation, md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    register(name, impl, md)(hrt, hru, hrv, new HailRep[Annotation] {
      def typ = t
    })
  }

  def registerAnn[T, U, V, W](name: String, t: TStruct, impl: (T, U, V, W) => Annotation, md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    register(name, impl, md)(hrt, hru, hrv, hrw, new HailRep[Annotation] {
      def typ = t
    })
  }

  def registerAggregator[T, U](name: String, ctor: () => TypedAggregator[U], md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, MethodType(hrt.typ), Arity0Aggregator[T, U](hru.typ, ctor), md)
  }

  def registerLambdaAggregator[T, U, V](name: String, ctor: ((Any) => Any) => TypedAggregator[V], md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, MethodType(hrt.typ, hru.typ), UnaryLambdaAggregator[T, U, V](hrv.typ, ctor), md)
  }

  def registerLambdaAggregator[T, U, V, W](name: String, ctor: ((Any) => Any, V) => TypedAggregator[W], md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    bind(name, MethodType(hrt.typ, hru.typ, hrv.typ), BinaryLambdaAggregator[T, U, V, W](hrw.typ, ctor), md)
  }

  def registerAggregator[T, U, V](name: String, ctor: (U) => TypedAggregator[V], md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, MethodType(hrt.typ, hru.typ), Arity1Aggregator[T, U, V](hrv.typ, ctor), md)
  }

  def registerAggregator[T, U, V, W, X](name: String, ctor: (U, V, W) => TypedAggregator[X], md: FunctionMetadata)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W], hrx: HailRep[X]) = {
    bind(name, MethodType(hrt.typ, hru.typ, hrv.typ, hrw.typ), Arity3Aggregator[T, U, V, W, X](hrx.typ, ctor), md)
  }

  val TT = TVariable("T")
  val TU = TVariable("U")
  val TV = TVariable("V")

  val TTHr = new HailRep[Any] {
    def typ = TT
  }
  val TUHr = new HailRep[Any] {
    def typ = TU
  }
  val TVHr = new HailRep[Any] {
    def typ = TV
  }

  registerOptionMethod("gt", { (x: Genotype) => x.gt }, FunctionMetadata())
  registerOptionMethod("gtj", { (x: Genotype) => x.gt.map(gtx => Genotype.gtPair(gtx).j) }, FunctionMetadata())
  registerOptionMethod("gtk", { (x: Genotype) => x.gt.map(gtx => Genotype.gtPair(gtx).k) }, FunctionMetadata())
  registerOptionMethod("ad", { (x: Genotype) => x.ad.map(a => a: IndexedSeq[Int]) }, FunctionMetadata())
  registerOptionMethod("dp", { (x: Genotype) => x.dp }, FunctionMetadata())
  registerOptionMethod("od", { (x: Genotype) => x.od }, FunctionMetadata())
  registerOptionMethod("gq", { (x: Genotype) => x.gq }, FunctionMetadata())
  registerOptionMethod("pl", { (x: Genotype) => x.pl.map(a => a: IndexedSeq[Int]) }, FunctionMetadata())
  registerOptionMethod("dosage", { (x: Genotype) => x.dosage.map(a => a: IndexedSeq[Double]) }, FunctionMetadata())
  registerMethod("isHomRef", { (x: Genotype) => x.isHomRef }, FunctionMetadata())
  registerMethod("isHet", { (x: Genotype) => x.isHet }, FunctionMetadata())
  registerMethod("isHomVar", { (x: Genotype) => x.isHomVar }, FunctionMetadata())
  registerMethod("isCalledNonRef", { (x: Genotype) => x.isCalledNonRef }, FunctionMetadata())
  registerMethod("isHetNonRef", { (x: Genotype) => x.isHetNonRef }, FunctionMetadata())
  registerMethod("isHetRef", { (x: Genotype) => x.isHetRef }, FunctionMetadata())
  registerMethod("isCalled", { (x: Genotype) => x.isCalled }, FunctionMetadata())
  registerMethod("isNotCalled", { (x: Genotype) => x.isNotCalled }, FunctionMetadata())
  registerOptionMethod("nNonRefAlleles", { (x: Genotype) => x.nNonRefAlleles }, FunctionMetadata())
  registerOptionMethod("pAB", { (x: Genotype) => x.pAB() }, FunctionMetadata())
  registerOptionMethod("fractionReadsRef", { (x: Genotype) => x.fractionReadsRef() }, FunctionMetadata())
  registerMethod("fakeRef", { (x: Genotype) => x.fakeRef }, FunctionMetadata())
  registerMethod("isDosage", { (x: Genotype) => x.isDosage }, FunctionMetadata())
  registerMethod("contig", { (x: Variant) => x.contig }, FunctionMetadata())
  registerMethod("start", { (x: Variant) => x.start }, FunctionMetadata())
  registerMethod("ref", { (x: Variant) => x.ref }, FunctionMetadata())
  registerMethod("altAlleles", { (x: Variant) => x.altAlleles }, FunctionMetadata())
  registerMethod("nAltAlleles", { (x: Variant) => x.nAltAlleles }, FunctionMetadata())
  registerMethod("nAlleles", { (x: Variant) => x.nAlleles }, FunctionMetadata())
  registerMethod("isBiallelic", { (x: Variant) => x.isBiallelic }, FunctionMetadata())
  registerMethod("nGenotypes", { (x: Variant) => x.nGenotypes }, FunctionMetadata())
  registerMethod("inXPar", { (x: Variant) => x.inXPar }, FunctionMetadata())
  registerMethod("inYPar", { (x: Variant) => x.inYPar }, FunctionMetadata())
  registerMethod("inXNonPar", { (x: Variant) => x.inXNonPar }, FunctionMetadata())
  registerMethod("inYNonPar", { (x: Variant) => x.inYNonPar }, FunctionMetadata())
  // assumes biallelic
  registerMethod("alt", { (x: Variant) => x.alt }, FunctionMetadata())
  registerMethod("altAllele", { (x: Variant) => x.altAllele }, FunctionMetadata())
  registerMethod("locus", { (x: Variant) => x.locus }, FunctionMetadata())
  registerMethod("contig", { (x: Locus) => x.contig }, FunctionMetadata())
  registerMethod("position", { (x: Locus) => x.position }, FunctionMetadata())
  registerMethod("start", { (x: Interval[Locus]) => x.start }, FunctionMetadata())
  registerMethod("end", { (x: Interval[Locus]) => x.end }, FunctionMetadata())
  registerMethod("ref", { (x: AltAllele) => x.ref }, FunctionMetadata())
  registerMethod("alt", { (x: AltAllele) => x.alt }, FunctionMetadata())
  registerMethod("isSNP", { (x: AltAllele) => x.isSNP }, FunctionMetadata())
  registerMethod("isMNP", { (x: AltAllele) => x.isMNP }, FunctionMetadata())
  registerMethod("isIndel", { (x: AltAllele) => x.isIndel }, FunctionMetadata())
  registerMethod("isInsertion", { (x: AltAllele) => x.isInsertion }, FunctionMetadata())
  registerMethod("isDeletion", { (x: AltAllele) => x.isDeletion }, FunctionMetadata())
  registerMethod("isComplex", { (x: AltAllele) => x.isComplex }, FunctionMetadata())
  registerMethod("isTransition", { (x: AltAllele) => x.isTransition }, FunctionMetadata())
  registerMethod("isTransversion", { (x: AltAllele) => x.isTransversion }, FunctionMetadata())
  registerMethod("isAutosomal", { (x: Variant) => x.isAutosomal }, FunctionMetadata())

  registerMethod("length", { (x: String) => x.length }, FunctionMetadata())

  registerUnaryNAFilteredCollectionMethod("sum", { (x: TraversableOnce[Int]) => x.sum }, FunctionMetadata())
  registerUnaryNAFilteredCollectionMethod("sum", { (x: TraversableOnce[Long]) => x.sum }, FunctionMetadata())
  registerUnaryNAFilteredCollectionMethod("sum", { (x: TraversableOnce[Float]) => x.sum }, FunctionMetadata())
  registerUnaryNAFilteredCollectionMethod("sum", { (x: TraversableOnce[Double]) => x.sum }, FunctionMetadata())

  registerUnaryNAFilteredCollectionMethod("min", { (x: TraversableOnce[Int]) => x.min }, FunctionMetadata())
  registerUnaryNAFilteredCollectionMethod("min", { (x: TraversableOnce[Long]) => x.min }, FunctionMetadata())
  registerUnaryNAFilteredCollectionMethod("min", { (x: TraversableOnce[Float]) => x.min }, FunctionMetadata())
  registerUnaryNAFilteredCollectionMethod("min", { (x: TraversableOnce[Double]) => x.min }, FunctionMetadata())

  registerUnaryNAFilteredCollectionMethod("max", { (x: TraversableOnce[Int]) => x.max }, FunctionMetadata())
  registerUnaryNAFilteredCollectionMethod("max", { (x: TraversableOnce[Long]) => x.max }, FunctionMetadata())
  registerUnaryNAFilteredCollectionMethod("max", { (x: TraversableOnce[Float]) => x.max }, FunctionMetadata())
  registerUnaryNAFilteredCollectionMethod("max", { (x: TraversableOnce[Double]) => x.max }, FunctionMetadata())

  registerUnaryNAFilteredCollectionMethod("median", { (x: TraversableOnce[Int]) => breeze.stats.median(DenseVector(x.toArray)) }, FunctionMetadata())
  registerUnaryNAFilteredCollectionMethod("median", { (x: TraversableOnce[Long]) => breeze.stats.median(DenseVector(x.toArray)) }, FunctionMetadata())
  registerUnaryNAFilteredCollectionMethod("median", { (x: TraversableOnce[Float]) => breeze.stats.median(DenseVector(x.toArray)) }, FunctionMetadata())
  registerUnaryNAFilteredCollectionMethod("median", { (x: TraversableOnce[Double]) => breeze.stats.median(DenseVector(x.toArray)) }, FunctionMetadata())

  registerUnaryNAFilteredCollectionMethod("mean", { (x: TraversableOnce[Int]) => x.sum / x.size.toDouble }, FunctionMetadata())
  registerUnaryNAFilteredCollectionMethod("mean", { (x: TraversableOnce[Long]) => x.sum / x.size.toDouble }, FunctionMetadata())
  registerUnaryNAFilteredCollectionMethod("mean", { (x: TraversableOnce[Float]) => x.sum / x.size.toDouble }, FunctionMetadata())
  registerUnaryNAFilteredCollectionMethod("mean", { (x: TraversableOnce[Double]) => x.sum / x.size }, FunctionMetadata())

  register("range", { (x: Int) => 0 until x: IndexedSeq[Int] }, FunctionMetadata())
  register("range", { (x: Int, y: Int) => x until y: IndexedSeq[Int] }, FunctionMetadata())
  register("range", { (x: Int, y: Int, step: Int) => x until y by step: IndexedSeq[Int] }, FunctionMetadata())
  register("Variant", { (x: String) =>
    val Array(chr, pos, ref, alts) = x.split(":")
    Variant(chr, pos.toInt, ref, alts.split(","))
  }, FunctionMetadata())
  register("Variant", { (x: String, y: Int, z: String, a: String) => Variant(x, y, z, a) }, FunctionMetadata())
  register("Variant", { (x: String, y: Int, z: String, a: IndexedSeq[String]) => Variant(x, y, z, a.toArray) }, FunctionMetadata())

  register("Locus", { (x: String) =>
    val Array(chr, pos) = x.split(":")
    Locus(chr, pos.toInt)
  }, FunctionMetadata())
  register("Locus", { (x: String, y: Int) => Locus(x, y) }, FunctionMetadata())
  register("Interval", { (x: Locus, y: Locus) => Interval(x, y) }, FunctionMetadata())
  registerAnn("hwe", TStruct(("rExpectedHetFrequency", TDouble), ("pHWE", TDouble)), { (nHomRef: Int, nHet: Int, nHomVar: Int) =>
    if (nHomRef < 0 || nHet < 0 || nHomVar < 0)
      fatal(s"got invalid (negative) argument to function `hwe': hwe($nHomRef, $nHet, $nHomVar)")
    val n = nHomRef + nHet + nHomVar
    val nAB = nHet
    val nA = nAB + 2 * nHomRef.min(nHomVar)

    val LH = LeveneHaldane(n, nA)
    Annotation(divOption(LH.getNumericalMean, n).orNull, LH.exactMidP(nAB))
  }, FunctionMetadata())
  registerAnn("fet", TStruct(("pValue", TDouble), ("oddsRatio", TDouble), ("ci95Lower", TDouble), ("ci95Upper", TDouble)), { (c1: Int, c2: Int, c3: Int, c4: Int) =>
    if (c1 < 0 || c2 < 0 || c3 < 0 || c4 < 0)
      fatal(s"got invalid argument to function `fet': fet($c1, $c2, $c3, $c4)")
    val fet = FisherExactTest(c1, c2, c3, c4)
    Annotation(fet(0).orNull, fet(1).orNull, fet(2).orNull, fet(3).orNull)
  }, FunctionMetadata())
  // NB: merge takes two structs, how do I deal with structs?
  register("exp", { (x: Double) => math.exp(x) }, FunctionMetadata())
  register("log10", { (x: Double) => math.log10(x) }, FunctionMetadata())
  register("sqrt", { (x: Double) => math.sqrt(x) }, FunctionMetadata())
  register("log", (x: Double) => math.log(x), FunctionMetadata())
  register("log", (x: Double, b: Double) => math.log(x) / math.log(b), FunctionMetadata())
  register("pow", (b: Double, x: Double) => math.pow(b, x), FunctionMetadata())

  register("pcoin", { (p: Double) => math.random < p }, FunctionMetadata())
  register("runif", { (min: Double, max: Double) => min + (max - min) * math.random }, FunctionMetadata())
  register("rnorm", { (mean: Double, sd: Double) => mean + sd * scala.util.Random.nextGaussian() }, FunctionMetadata())

  register("pnorm", { (x: Double) => pnorm(x) }, FunctionMetadata())
  register("qnorm", { (p: Double) => qnorm(p) }, FunctionMetadata())

  register("pchisq1tail", { (x: Double) => chiSquaredTail(1.0, x) }, FunctionMetadata())
  register("qchisq1tail", { (p: Double) => inverseChiSquaredTail(1.0, p) }, FunctionMetadata())

  register("!", (a: Boolean) => !a, FunctionMetadata())

  registerConversion((x: Int) => x.toDouble, priority = 2)
  registerConversion { (x: Long) => x.toDouble }
  registerConversion { (x: Int) => x.toLong }
  registerConversion { (x: Float) => x.toDouble }

  registerConversion((x: IndexedSeq[Any]) => x.map { xi =>
    if (xi == null)
      null
    else
      xi.asInstanceOf[Int].toDouble
  }, priority = 2)(arrayHr(boxedintHr), arrayHr(boxeddoubleHr))

  registerConversion((x: IndexedSeq[Any]) => x.map { xi =>
    if (xi == null)
      null
    else
      xi.asInstanceOf[Long].toDouble
  })(arrayHr(boxedlongHr), arrayHr(boxeddoubleHr))

  registerConversion((x: IndexedSeq[Any]) => x.map { xi =>
    if (xi == null)
      null
    else
      xi.asInstanceOf[Int].toLong
  })(arrayHr(boxedintHr), arrayHr(boxedlongHr))

  registerConversion((x: IndexedSeq[Any]) => x.map { xi =>
    if (xi == null)
      null
    else
      xi.asInstanceOf[Float].toDouble
  })(arrayHr(boxedfloatHr), arrayHr(boxeddoubleHr))

  register("gtj", (i: Int) => Genotype.gtPair(i).j, FunctionMetadata())
  register("gtk", (i: Int) => Genotype.gtPair(i).k, FunctionMetadata())
  register("gtIndex", (j: Int, k: Int) => Genotype.gtIndex(j, k), FunctionMetadata())

  registerConversion((x: Any) =>
    if (x != null)
      x.asInstanceOf[Int].toDouble
    else
      null, priority = 2)(aggregableHr(boxedintHr), aggregableHr(boxeddoubleHr))
  registerConversion { (x: Any) =>
    if (x != null)
      x.asInstanceOf[Long].toDouble
    else
      null
  }(aggregableHr(boxedlongHr), aggregableHr(boxeddoubleHr))

  registerConversion { (x: Any) =>
    if (x != null)
      x.asInstanceOf[Int].toLong
    else
      null
  }(aggregableHr(boxedintHr), aggregableHr(boxedlongHr))

  registerConversion { (x: Any) =>
    if (x != null)
      x.asInstanceOf[Float].toDouble
    else
      null
  }(aggregableHr(boxedfloatHr), aggregableHr(boxeddoubleHr))

  registerMethod("split", (s: String, p: String) => s.split(p): IndexedSeq[String], FunctionMetadata())

  registerMethod("oneHotAlleles", (g: Genotype, v: Variant) => g.oneHotAlleles(v).orNull, FunctionMetadata())

  registerMethod("oneHotGenotype", (g: Genotype, v: Variant) => g.oneHotGenotype(v).orNull, FunctionMetadata())

  registerMethod("replace", (str: String, pattern1: String, pattern2: String) =>
    str.replaceAll(pattern1, pattern2), FunctionMetadata())

  registerMethod("contains", (interval: Interval[Locus], locus: Locus) => interval.contains(locus), FunctionMetadata())

  registerMethod("length", (a: IndexedSeq[Any]) => a.length, FunctionMetadata())(arrayHr(TTHr), intHr)
  registerMethod("size", (a: IndexedSeq[Any]) => a.size, FunctionMetadata())(arrayHr(TTHr), intHr)
  registerMethod("size", (s: Set[Any]) => s.size, FunctionMetadata())(setHr(TTHr), intHr)
  registerMethod("size", (d: Map[String, Any]) => d.size, FunctionMetadata())(dictHr(TTHr), intHr)

  registerMethod("id", (s: String) => s, FunctionMetadata())(sampleHr, stringHr)

  registerMethod("isEmpty", (a: IndexedSeq[Any]) => a.isEmpty, FunctionMetadata())(arrayHr(TTHr), boolHr)
  registerMethod("isEmpty", (s: Set[Any]) => s.isEmpty, FunctionMetadata())(setHr(TTHr), boolHr)
  registerMethod("isEmpty", (d: Map[String, Any]) => d.isEmpty, FunctionMetadata())(dictHr(TTHr), boolHr)

  registerMethod("toSet", (a: IndexedSeq[Any]) => a.toSet, FunctionMetadata())(arrayHr(TTHr), setHr(TTHr))
  registerMethod("toSet", (a: Set[Any]) => a, FunctionMetadata())(setHr(TTHr), setHr(TTHr))
  registerMethod("toArray", (a: Set[Any]) => a.toArray[Any]: IndexedSeq[Any], FunctionMetadata())(setHr(TTHr), arrayHr(TTHr))
  registerMethod("toArray", (a: IndexedSeq[Any]) => a, FunctionMetadata())(arrayHr(TTHr), arrayHr(TTHr))

  registerMethod("head", (a: IndexedSeq[Any]) => a.head, FunctionMetadata())(arrayHr(TTHr), TTHr)
  registerMethod("tail", (a: IndexedSeq[Any]) => a.tail, FunctionMetadata())(arrayHr(TTHr), arrayHr(TTHr))

  registerMethod("head", (a: Set[Any]) => a.head, FunctionMetadata())(setHr(TTHr), TTHr)
  registerMethod("tail", (a: Set[Any]) => a.tail, FunctionMetadata())(setHr(TTHr), setHr(TTHr))

  registerMethod("flatten", (a: IndexedSeq[IndexedSeq[Any]]) =>
    flattenOrNull[IndexedSeq, Any](IndexedSeq.newBuilder[Any], a), FunctionMetadata()
  )(arrayHr(arrayHr(TTHr)), arrayHr(TTHr))

  registerMethod("flatten", (s: Set[Set[Any]]) =>
    flattenOrNull[Set, Any](Set.newBuilder[Any], s), FunctionMetadata()
  )(setHr(setHr(TTHr)), setHr(TTHr))

  registerMethod("mkString", (a: IndexedSeq[String], d: String) => a.mkString(d), FunctionMetadata())(
    arrayHr(stringHr), stringHr, stringHr)
  registerMethod("mkString", (s: Set[String], d: String) => s.mkString(d), FunctionMetadata())(
    setHr(stringHr), stringHr, stringHr)

  registerMethod("contains", (s: Set[Any], x: Any) => s.contains(x), FunctionMetadata())(setHr(TTHr), TTHr, boolHr)
  registerMethod("contains", (d: Map[String, Any], x: String) => d.contains(x), FunctionMetadata())(dictHr(TTHr), stringHr, boolHr)

  registerLambdaMethod("find", (a: IndexedSeq[Any], f: (Any) => Any) =>
    a.find { elt =>
      val r = f(elt)
      r != null && r.asInstanceOf[Boolean]
    }.orNull, FunctionMetadata()
  )(arrayHr(TTHr), unaryHr(TTHr, boolHr), TTHr)

  registerLambdaMethod("find", (s: Set[Any], f: (Any) => Any) =>
    s.find { elt =>
      val r = f(elt)
      r != null && r.asInstanceOf[Boolean]
    }.orNull, FunctionMetadata()
  )(setHr(TTHr), unaryHr(TTHr, boolHr), TTHr)

  registerLambdaMethod("map", (a: IndexedSeq[Any], f: (Any) => Any) =>
    a.map(f), FunctionMetadata()
  )(arrayHr(TTHr), unaryHr(TTHr, TUHr), arrayHr(TUHr))

  registerLambdaMethod("map", (s: Set[Any], f: (Any) => Any) =>
    s.map(f), FunctionMetadata()
  )(setHr(TTHr), unaryHr(TTHr, TUHr), setHr(TUHr))

  registerLambdaMethod("mapValues", (a: Map[String, Any], f: (Any) => Any) =>
    a.map { case (k, v) => (k, f(v)) }, FunctionMetadata()
  )(dictHr(TTHr), unaryHr(TTHr, TUHr), dictHr(TUHr))

  registerLambdaMethod("flatMap", (a: IndexedSeq[Any], f: (Any) => Any) =>
    flattenOrNull[IndexedSeq, Any](IndexedSeq.newBuilder[Any],
      a.map(f).asInstanceOf[IndexedSeq[IndexedSeq[Any]]]), FunctionMetadata()
  )(arrayHr(TTHr), unaryHr(TTHr, arrayHr(TUHr)), arrayHr(TUHr))

  registerLambdaMethod("flatMap", (s: Set[Any], f: (Any) => Any) =>
    flattenOrNull[Set, Any](Set.newBuilder[Any],
      s.map(f).asInstanceOf[Set[Set[Any]]]), FunctionMetadata()
  )(setHr(TTHr), unaryHr(TTHr, setHr(TUHr)), setHr(TUHr))

  registerLambdaMethod("exists", (a: IndexedSeq[Any], f: (Any) => Any) =>
    a.exists { x =>
      val r = f(x)
      r != null && r.asInstanceOf[Boolean]
    }, FunctionMetadata()
  )(arrayHr(TTHr), unaryHr(TTHr, boolHr), boolHr)

  registerLambdaMethod("exists", (s: Set[Any], f: (Any) => Any) =>
    s.exists { x =>
      val r = f(x)
      r != null && r.asInstanceOf[Boolean]
    }, FunctionMetadata()
  )(setHr(TTHr), unaryHr(TTHr, boolHr), boolHr)

  registerLambdaMethod("forall", (a: IndexedSeq[Any], f: (Any) => Any) =>
    a.forall { x =>
      val r = f(x)
      r != null && r.asInstanceOf[Boolean]
    }, FunctionMetadata()
  )(arrayHr(TTHr), unaryHr(TTHr, boolHr), boolHr)

  registerLambdaMethod("forall", (s: Set[Any], f: (Any) => Any) =>
    s.forall { x =>
      val r = f(x)
      r != null && r.asInstanceOf[Boolean]
    }, FunctionMetadata()
  )(setHr(TTHr), unaryHr(TTHr, boolHr), boolHr)

  registerLambdaMethod("filter", (a: IndexedSeq[Any], f: (Any) => Any) =>
    a.filter { x =>
      val r = f(x)
      r != null && r.asInstanceOf[Boolean]
    }, FunctionMetadata()
  )(arrayHr(TTHr), unaryHr(TTHr, boolHr), arrayHr(TTHr))

  registerLambdaMethod("filter", (s: Set[Any], f: (Any) => Any) =>
    s.filter { x =>
      val r = f(x)
      r != null && r.asInstanceOf[Boolean]
    }, FunctionMetadata()
  )(setHr(TTHr), unaryHr(TTHr, boolHr), setHr(TTHr))

  registerAggregator[Any, Long]("count", () => new CountAggregator(), FunctionMetadata())(aggregableHr(TTHr), longHr)

  registerAggregator[Any, IndexedSeq[Any]]("collect", () => new CollectAggregator(), FunctionMetadata())(aggregableHr(TTHr), arrayHr(TTHr))

  registerAggregator[Int, Int]("sum", () => new SumAggregator[Int](), FunctionMetadata())(aggregableHr(intHr), intHr)

  registerAggregator[Long, Long]("sum", () => new SumAggregator[Long](), FunctionMetadata())(aggregableHr(longHr), longHr)

  registerAggregator[Float, Float]("sum", () => new SumAggregator[Float](), FunctionMetadata())(aggregableHr(floatHr), floatHr)

  registerAggregator[Double, Double]("sum", () => new SumAggregator[Double](), FunctionMetadata())(aggregableHr(doubleHr), doubleHr)

  registerAggregator[IndexedSeq[Int], IndexedSeq[Int]]("sum", () => new SumArrayAggregator[Int](), FunctionMetadata())(aggregableHr(arrayHr(intHr)), arrayHr(intHr))

  registerAggregator[IndexedSeq[Long], IndexedSeq[Long]]("sum", () => new SumArrayAggregator[Long](), FunctionMetadata())(aggregableHr(arrayHr(longHr)), arrayHr(longHr))

  registerAggregator[IndexedSeq[Float], IndexedSeq[Float]]("sum", () => new SumArrayAggregator[Float](), FunctionMetadata())(aggregableHr(arrayHr(floatHr)), arrayHr(floatHr))

  registerAggregator[IndexedSeq[Double], IndexedSeq[Double]]("sum", () => new SumArrayAggregator[Double](), FunctionMetadata())(aggregableHr(arrayHr(doubleHr)), arrayHr(doubleHr))

  registerAggregator[Genotype, Any]("infoScore", () => new InfoScoreAggregator(), FunctionMetadata())(aggregableHr(genotypeHr),
    new HailRep[Any] {
      def typ = InfoScoreCombiner.signature
    })

  registerAggregator[Genotype, Any]("hardyWeinberg", () => new HWEAggregator(), FunctionMetadata())(aggregableHr(genotypeHr),
    new HailRep[Any] {
      def typ = HWECombiner.signature
    })

  registerAggregator[Any, Any]("counter", () => new CounterAggregator(), FunctionMetadata())(aggregableHr(TTHr),
    new HailRep[Any] {
      def typ = TArray(TStruct("key" -> TTHr.typ, "count" -> TLong))
    })

  registerAggregator[Double, Any]("stats", () => new StatAggregator(), FunctionMetadata())(aggregableHr(doubleHr),
    new HailRep[Any] {
      def typ = TStruct(("mean", TDouble), ("stdev", TDouble), ("min", TDouble),
        ("max", TDouble), ("nNotMissing", TLong), ("sum", TDouble))
    })

  registerAggregator[Double, Double, Double, Int, Any]("hist", (start: Double, end: Double, bins: Int) => {
    if (bins <= 0)
      fatal(s"""method `hist' expects `bins' argument to be > 0, but got $bins""")

    val binSize = (end - start) / bins
    if (binSize <= 0)
      fatal(
        s"""invalid bin size from given arguments (start = $start, end = $end, bins = $bins)
            |  Method requires positive bin size [(end - start) / bins], but got ${ binSize.formatted("%.2f") }
                  """.stripMargin)

    val indices = Array.tabulate(bins + 1)(i => start + i * binSize)

    new HistAggregator(indices)
  }, FunctionMetadata())(aggregableHr(doubleHr), doubleHr, doubleHr, intHr, new HailRep[Any] {
    def typ = HistogramCombiner.schema
  })

  registerLambdaAggregator[Genotype, (Any) => Any, Any]("callStats", (vf: (Any) => Any) => new CallStatsAggregator(vf), FunctionMetadata())(
    aggregableHr(genotypeHr), unaryHr(genotypeHr, variantHr), new HailRep[Any] {
      def typ = CallStats.schema
    })

  registerLambdaAggregator[Genotype, (Any) => Any, Any]("inbreeding", (af: (Any) => Any) => new InbreedingAggregator(af), FunctionMetadata())(
    aggregableHr(genotypeHr), unaryHr(genotypeHr, doubleHr), new HailRep[Any] {
      def typ = InbreedingCombiner.signature
    })

  registerLambdaAggregator[Any, (Any) => Any, Any]("fraction", (f: (Any) => Any) => new FractionAggregator(f), FunctionMetadata())(
    aggregableHr(TTHr), unaryHr(TTHr, boxedboolHr), boxeddoubleHr)

  registerAggregator("take", (n: Int) => new TakeAggregator(n), FunctionMetadata())(
    aggregableHr(TTHr), intHr, arrayHr(TTHr))

  registerLambdaAggregator("takeBy", (f: (Any) => Any, n: Int) => new TakeByAggregator[Int](f, n), FunctionMetadata())(
    aggregableHr(TTHr), unaryHr(TTHr, boxedintHr), intHr, arrayHr(TTHr))
  registerLambdaAggregator("takeBy", (f: (Any) => Any, n: Int) => new TakeByAggregator[Long](f, n), FunctionMetadata())(
    aggregableHr(TTHr), unaryHr(TTHr, boxedlongHr), intHr, arrayHr(TTHr))
  registerLambdaAggregator("takeBy", (f: (Any) => Any, n: Int) => new TakeByAggregator[Float](f, n), FunctionMetadata())(
    aggregableHr(TTHr), unaryHr(TTHr, boxedfloatHr), intHr, arrayHr(TTHr))
  registerLambdaAggregator("takeBy", (f: (Any) => Any, n: Int) => new TakeByAggregator[Double](f, n), FunctionMetadata())(
    aggregableHr(TTHr), unaryHr(TTHr, boxeddoubleHr), intHr, arrayHr(TTHr))
  registerLambdaAggregator("takeBy", (f: (Any) => Any, n: Int) => new TakeByAggregator[String](f, n), FunctionMetadata())(
    aggregableHr(TTHr), unaryHr(TTHr, stringHr), intHr, arrayHr(TTHr))

  val aggST = Box[SymbolTable]()

  registerLambdaAggregatorTransformer("flatMap", { (a: CPS[Any], f: (Any) => Any) =>
    { (k: Any => Any) => a { x =>
      val r = f(x).asInstanceOf[IndexedSeq[Any]]
      var i = 0
      while (i < r.size) {
        k(r(i))
        i += 1
      }
    } }
  }, FunctionMetadata())(aggregableHr(TTHr, aggST), unaryHr(TTHr, arrayHr(TUHr)), aggregableHr(TUHr, aggST))

  registerLambdaAggregatorTransformer("flatMap", { (a: CPS[Any], f: (Any) => Any) =>
    { (k: Any => Any) => a { x => f(x).asInstanceOf[Set[Any]].foreach(k) } }
  }, FunctionMetadata())(aggregableHr(TTHr, aggST), unaryHr(TTHr, setHr(TUHr)), aggregableHr(TUHr, aggST))

  registerLambdaAggregatorTransformer("filter", { (a: CPS[Any], f: (Any) => Any) =>
    { (k: Any => Any) => a { x =>
      val r = f(x)
      if (r != null && r.asInstanceOf[Boolean])
        k(x)
    } }
  }, FunctionMetadata())(aggregableHr(TTHr, aggST), unaryHr(TTHr, boolHr), aggregableHr(TTHr, aggST))

  registerLambdaAggregatorTransformer("map", { (a: CPS[Any], f: (Any) => Any) =>
    { (k: Any => Any) => a { x => k(f(x)) } }
  }, FunctionMetadata())(aggregableHr(TTHr, aggST), unaryHr(TTHr, TUHr), aggregableHr(TUHr, aggST))

  type Id[T] = T

  def registerNumeric[T, S](name: String, f: (T, T) => S)(implicit hrt: HailRep[T], hrs: HailRep[S]) {
    val hrboxedt = new HailRep[Any] {
      def typ: Type = hrt.typ
    }
    val hrboxeds = new HailRep[Any] {
      def typ: Type = hrs.typ
    }

    register(name, f, FunctionMetadata())

    register(name, (x: IndexedSeq[Any], y: T) =>
      x.map { xi =>
        if (xi == null)
          null
        else
          f(xi.asInstanceOf[T], y)
      }, FunctionMetadata())(arrayHr(hrboxedt), hrt, arrayHr(hrboxeds))

    register(name, (x: T, y: IndexedSeq[Any]) => y.map { yi =>
      if (yi == null)
        null
      else
        f(x, yi.asInstanceOf[T])
    }, FunctionMetadata())(hrt, arrayHr(hrboxedt), arrayHr(hrboxeds))

    register(name, { (x: IndexedSeq[Any], y: IndexedSeq[Any]) =>
      if (x.length != y.length) fatal(
        s"""Cannot apply operation $name to arrays of unequal length:
            |  Left: ${ x.length } elements
            |  Right: ${ y.length } elements""".stripMargin)
      (x, y).zipped.map { case (xi, yi) =>
        if (xi == null || yi == null)
          null
        else
          f(xi.asInstanceOf[T], yi.asInstanceOf[T])
      }
    }, FunctionMetadata())(arrayHr(hrboxedt), arrayHr(hrboxedt), arrayHr(hrboxeds))
  }

  registerMethod("toInt", (s: String) => s.toInt, FunctionMetadata())
  registerMethod("toLong", (s: String) => s.toLong, FunctionMetadata())
  registerMethod("toFloat", (s: String) => s.toFloat, FunctionMetadata())
  registerMethod("toDouble", (s: String) => s.toDouble, FunctionMetadata())

  registerMethod("toInt", (b: Boolean) => b.toInt, FunctionMetadata())
  registerMethod("toLong", (b: Boolean) => b.toLong, FunctionMetadata())
  registerMethod("toFloat", (b: Boolean) => b.toFloat, FunctionMetadata())
  registerMethod("toDouble", (b: Boolean) => b.toDouble, FunctionMetadata())

  def registerNumericType[T]()(implicit ev: Numeric[T], hrt: HailRep[T]) {
    registerNumeric("+", ev.plus)
    registerNumeric("-", ev.minus)
    registerNumeric("*", ev.times)
    registerNumeric("/", (x: T, y: T) => ev.toDouble(x) / ev.toDouble(y))

    registerMethod("abs", ev.abs _, FunctionMetadata())
    registerMethod("signum", ev.signum _, FunctionMetadata())

    register("-", ev.negate _, FunctionMetadata())
    register("fromInt", ev.fromInt _, FunctionMetadata())

    registerMethod("toInt", ev.toInt _, FunctionMetadata())
    registerMethod("toLong", ev.toLong _, FunctionMetadata())
    registerMethod("toFloat", ev.toFloat _, FunctionMetadata())
    registerMethod("toDouble", ev.toDouble _, FunctionMetadata())
  }

  registerNumericType[Int]()
  registerNumericType[Long]()
  registerNumericType[Float]()
  registerNumericType[Double]()

  register("==", (a: Any, b: Any) => a == b, FunctionMetadata())(TTHr, TUHr, boolHr)
  register("!=", (a: Any, b: Any) => a != b, FunctionMetadata())(TTHr, TUHr, boolHr)

  def registerOrderedType[T]()(implicit ord: Ordering[T], hrt: HailRep[T]) {
    val hrboxedt = new HailRep[Any] {
      def typ: Type = hrt.typ
    }

    register("<", ord.lt _, FunctionMetadata())
    register("<=", ord.lteq _, FunctionMetadata())
    register(">", ord.gt _, FunctionMetadata())
    register(">=", ord.gteq _, FunctionMetadata())

    registerMethod("min", ord.min _, FunctionMetadata())
    registerMethod("max", ord.max _, FunctionMetadata())

    registerMethod("sort", (a: IndexedSeq[Any]) => a.sorted(extendOrderingToNull(ord)), FunctionMetadata())(arrayHr(hrboxedt), arrayHr(hrboxedt))
    registerMethod("sort", (a: IndexedSeq[Any], ascending: Boolean) =>
      a.sorted(extendOrderingToNull(
        if (ascending)
          ord
        else
          ord.reverse)), FunctionMetadata()
    )(arrayHr(hrboxedt), boolHr, arrayHr(hrboxedt))

    registerLambdaMethod("sortBy", (a: IndexedSeq[Any], f: (Any) => Any) =>
      a.sortBy(f)(extendOrderingToNull(ord)), FunctionMetadata()
    )(arrayHr(TTHr), unaryHr(TTHr, hrboxedt), arrayHr(TTHr))

    registerLambdaMethod("sortBy", (a: IndexedSeq[Any], f: (Any) => Any, ascending: Boolean) =>
      a.sortBy(f)(extendOrderingToNull(
        if (ascending)
          ord
        else
          ord.reverse)), FunctionMetadata()
    )(arrayHr(TTHr), unaryHr(TTHr, hrboxedt), boolHr, arrayHr(TTHr))
  }

  registerOrderedType[Boolean]()
  registerOrderedType[Int]()
  registerOrderedType[Long]()
  registerOrderedType[Float]()
  registerOrderedType[Double]()
  registerOrderedType[String]()

  registerMethod("//", (x: Int, y: Int) => java.lang.Math.floorDiv(x, y), FunctionMetadata())
  registerMethod("//", (x: Long, y: Long) => java.lang.Math.floorDiv(x, y), FunctionMetadata())
  registerMethod("//", (x: Float, y: Float) => math.floor(x / y).toFloat, FunctionMetadata())
  registerMethod("//", (x: Double, y: Double) => math.floor(x / y), FunctionMetadata())

  register("%", (x: Int, y: Int) => java.lang.Math.floorMod(x, y), FunctionMetadata())
  register("%", (x: Long, y: Long) => java.lang.Math.floorMod(x,y), FunctionMetadata())
  register("%", (x: Float, y: Float) => { val t = x % y; if (x >= 0 && y > 0 || x <= 0 && y < 0 || t == 0) t else t + y }, FunctionMetadata())
  register("%", (x: Double, y: Double) => { val t = x % y; if (x >= 0 && y > 0 || x <= 0 && y < 0 || t == 0) t else t + y }, FunctionMetadata())

  register("+", (x: String, y: Any) => x + y, FunctionMetadata())(stringHr, TTHr, stringHr)

  register("~", (s: String, t: String) => s.r.findFirstIn(t).isDefined, FunctionMetadata())

  registerSpecial("isMissing", (g: () => Any) => g() == null, FunctionMetadata())(TTHr, boolHr)
  registerSpecial("isDefined", (g: () => Any) => g() != null, FunctionMetadata())(TTHr, boolHr)

  registerSpecial("||", { (f1: () => Any, f2: () => Any) =>
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
  }, FunctionMetadata())(boolHr, boolHr, boxedboolHr)

  registerSpecial("&&", { (f1: () => Any, f2: () => Any) =>
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
  }, FunctionMetadata())(boolHr, boolHr, boxedboolHr)

  registerMethodSpecial("orElse", { (f1: () => Any, f2: () => Any) =>
    val v = f1()
    if (v == null)
      f2()
    else
      v
  }, FunctionMetadata())(TTHr, TTHr, TTHr)

  registerMethod("[]", (a: IndexedSeq[Any], i: Int) => if (i >= 0) a(i) else a(a.length + i), FunctionMetadata())(arrayHr(TTHr), intHr, TTHr)
  registerMethod("[]", (a: Map[String, Any], i: String) => a(i), FunctionMetadata())(dictHr(TTHr), stringHr, TTHr)
  registerMethod("[]", (a: String, i: Int) => (if (i >= 0) a(i) else a(a.length + i)).toString, FunctionMetadata())(stringHr, intHr, charHr)

  registerMethod("[:]", (a: IndexedSeq[Any]) => a, FunctionMetadata())(arrayHr(TTHr), arrayHr(TTHr))
  registerMethod("[*:]", (a: IndexedSeq[Any], i: Int) => a.slice(i, a.length), FunctionMetadata())(arrayHr(TTHr), intHr, arrayHr(TTHr))
  registerMethod("[:*]", (a: IndexedSeq[Any], i: Int) => a.slice(0, i), FunctionMetadata())(arrayHr(TTHr), intHr, arrayHr(TTHr))
  registerMethod("[*:*]", (a: IndexedSeq[Any], i: Int, j: Int) => a.slice(i, j), FunctionMetadata())(arrayHr(TTHr), intHr, intHr, arrayHr(TTHr))
}
