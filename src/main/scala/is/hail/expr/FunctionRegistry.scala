package is.hail.expr

import breeze.linalg.DenseVector
import is.hail.annotations.Annotation
import is.hail.asm4s._
import is.hail.expr.types._
import is.hail.methods._
import is.hail.stats._
import is.hail.utils.EitherIsAMonad._
import is.hail.utils._
import is.hail.variant._
import org.objectweb.asm.Opcodes._
import org.objectweb.asm.tree._

import scala.collection.generic.Growable
import scala.collection.mutable
import scala.language.higherKinds
import scala.reflect.ClassTag
import org.apache.commons.math3.special.Gamma
import org.apache.spark.sql.Row
import org.json4s.jackson.JsonMethods

object FunctionRegistry {

  sealed trait LookupError {
    def message: String
  }

  sealed case class NotFound(name: String, typ: TypeTag, rtTypExp: Option[Type] = None) extends LookupError {
    def message = {
      rtTypExp match {
        case None => s"No function found with name `$name' and argument ${ plural(typ.xs.size, "type") } $typ"
        case Some(rtTyp) => s"No function found with name `$name', argument ${ plural(typ.xs.size, "type") } $typ, and return type $rtTyp"
      }
    }
  }

  sealed case class Ambiguous(name: String, typ: TypeTag, alternates: Seq[(TypeTag, Fun)]) extends LookupError {
    def message =
      s"""found ${ alternates.size } ambiguous matches for $typ:
         |  ${ alternates.map(_._1).mkString("\n  ") }""".stripMargin
  }

  type Err[T] = Either[LookupError, T]

  private val registry = mutable.HashMap[String, Seq[(TypeTag, Fun)]]().withDefaultValue(Seq.empty)

  private def lookup(name: String, typ: TypeTag, rtTypConcrete: Option[Type] = None): Err[Fun] = {

    val matches = registry(name).flatMap { case (tt, f) =>
      tt.clear()
      f.retType.clear()

      if (tt.xs.size == typ.xs.size
        && (tt.xs, typ.xs).zipped.forall { (l, r) => l.unify(r) }
        && rtTypConcrete.forall(f.retType.unify(_)))
        Some((tt.subst(), f.captureType().subst()))
      else
        None
    }

    matches.headOption
      .toRight[LookupError](NotFound(name, typ, rtTypConcrete))
      .flatMap { it =>
        if (matches.size == 1)
          Right(it._2)
        else
          Left(Ambiguous(name, typ, matches))
      }
  }

  private def bind(name: String, typ: TypeTag, f: Fun) = {
    registry.updateValue(name, Seq.empty, (typ, f) +: _)
  }

  def getRegistry() = registry

  def lookupFieldReturnType(typ: Type, typs: Seq[Type], name: String): Err[Type] =
    lookup(name, FieldType(typ +: typs: _*)).map(-_.retType)

  def lookupMethodReturnType(typ: Type, typs: Seq[Type], name: String): Err[Type] =
    lookup(name, MethodType(typ +: typs: _*)).map(-_.retType)

  def lookupFunReturnType(name: String, typs: Seq[Type]): Err[Type] =
    lookup(name, FunType(typs: _*)).map(-_.retType)

  def registerField[T, U](name: String, impl: T => U)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, FieldType(hrt.typ), UnaryFun[T, U](hru.typ))
  }

  def registerFieldCode[T, U](name: String, impl: (Code[T]) => CM[Code[U]])
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, FieldType(hrt.typ), UnaryFunCode[T, U](hru.typ))
  }

  def registerMethod[T, U](name: String, impl: T => U)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, MethodType(hrt.typ), UnaryFun[T, U](hru.typ))
  }

  def registerMethodCode[T, U](name: String, impl: (Code[T]) => CM[Code[U]])
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, MethodType(hrt.typ), UnaryFunCode[T, U](hru.typ))
  }

  def registerMethodDependent[T, U](name: String, impl: () => T => U)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, MethodType(hrt.typ), UnaryDependentFun[T, U](hru.typ))
  }

  def registerMethodSpecial[T, U](name: String, impl: (() => Any) => U)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, MethodType(hrt.typ), UnarySpecial[T, U](hru.typ))
  }

  def registerMethod[T, U, V](name: String, impl: (T, U) => V)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, MethodType(hrt.typ, hru.typ), BinaryFun[T, U, V](hrv.typ))
  }

  def registerMethodDependent[T, U, V](name: String, impl: () => (T, U) => V)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, MethodType(hrt.typ, hru.typ), BinaryDependentFun[T, U, V](hrv.typ))
  }

  def registerMethodCode[T, U, V](name: String, impl: (Code[T], Code[U]) => CM[Code[V]])
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, MethodType(hrt.typ, hru.typ), BinaryFunCode[T, U, V](hrv.typ))
  }

  def registerMethodSpecial[T, U, V](name: String, impl: (() => Any, () => Any) => V)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, MethodType(hrt.typ, hru.typ), BinarySpecial[T, U, V](hrv.typ))
  }

  def registerLambdaMethod[T, U, V](name: String, impl: (T, (Any) => Any) => V)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    val m = BinaryLambdaFun[T, U, V](hrv.typ)
    bind(name, MethodType(hrt.typ, hru.typ), m)
  }

  def registerLambdaMethod[T, U, V, W](name: String, impl: (T, (Any) => Any, V) => W)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    val m = Arity3LambdaMethod[T, U, V, W](hrw.typ)
    bind(name, MethodType(hrt.typ, hru.typ, hrv.typ), m)
  }

  def registerLambda[T, U, V, W](name: String, impl: ((Any) => Any, U, V) => W)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    val m = Arity3LambdaFun[T, U, V, W](hrw.typ)
    bind(name, FunType(hrt.typ, hru.typ, hrv.typ), m)
  }

  def registerLambdaAggregatorTransformer[T, U, V](name: String, impl: (CPS[Any], (Any) => Any) => CPS[V],
    codeImpl: (Code[AnyRef], Code[AnyRef] => CM[Code[AnyRef]]) => CMCodeCPS[AnyRef])
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    val m = BinaryLambdaAggregatorTransformer[T, U, V](hrv.typ)
    bind(name, MethodType(hrt.typ, hru.typ), m)
  }

  def registerMethod[T, U, V, W](name: String, impl: (T, U, V) => W)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    bind(name, MethodType(hrt.typ, hru.typ, hrv.typ), Arity3Fun[T, U, V, W](hrw.typ))
  }

  def register[T, U](name: String, impl: T => U)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, FunType(hrt.typ), UnaryFun[T, U](hru.typ))
  }

  def registerCode[T, U](name: String, impl: Code[T] => CM[Code[U]])
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, FunType(hrt.typ), UnaryFunCode[T, U](hru.typ))
  }

  def registerDependentCode[T, U](name: String, impl: () => Code[T] => CM[Code[U]])
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, FunType(hrt.typ), UnaryDependentFunCode[T, U](hru.typ))
  }

  def registerDependent[T, U](name: String, impl: () => T => U)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, FunType(hrt.typ), UnaryDependentFun[T, U](hru.typ))
  }

  def registerSpecial[T, U](name: String, impl: (() => Any) => U)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, FunType(hrt.typ), UnarySpecial[T, U](hru.typ))
  }

  def registerUnaryNAFilteredCollectionMethod[T, U](name: String, impl: TraversableOnce[T] => U)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, MethodType(TArray(hrt.typ)), UnaryFun[IndexedSeq[_], U](hru.typ))
    bind(name, MethodType(TSet(hrt.typ)), UnaryFun[Set[_], U](hru.typ))
  }

  def register[T, U, V](name: String, impl: (T, U) => V)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, FunType(hrt.typ, hru.typ), BinaryFun[T, U, V](hrv.typ))
  }

  def registerCode[T, U, V](name: String, impl: (Code[T], Code[U]) => CM[Code[V]])
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, FunType(hrt.typ, hru.typ), BinaryFunCode[T, U, V](hrv.typ))
  }

  def registerSpecial[T, U, V](name: String, impl: (() => Any, () => Any) => V)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, FunType(hrt.typ, hru.typ), BinarySpecial[T, U, V](hrv.typ))
  }

  def registerSpecialCode[T, U, V](name: String, impl: (Code[T], Code[U]) => CM[Code[V]])
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, FunType(hrt.typ, hru.typ), BinarySpecialCode[T, U, V](hrv.typ))
  }

  def registerDependent[T, U, V](name: String, impl: () => (T, U) => V)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, FunType(hrt.typ, hru.typ), BinaryDependentFun[T, U, V](hrv.typ))
  }

  def register[T, U, V, W](name: String, impl: (T, U, V) => W)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    bind(name, FunType(hrt.typ, hru.typ, hrv.typ), Arity3Fun[T, U, V, W](hrw.typ))
  }

  def registerSpecial[T, U, V, W](name: String, impl: (() => Any, () => Any, () => Any) => W)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    bind(name, FunType(hrt.typ, hru.typ, hrv.typ), Arity3Special[T, U, V, W](hrw.typ))
  }

  def registerDependent[T, U, V, W](name: String, impl: () => (T, U, V) => W)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    bind(name, FunType(hrt.typ, hru.typ, hrv.typ), Arity3DependentFun[T, U, V, W](hrw.typ))
  }

  def register[T, U, V, W, X](name: String, impl: (T, U, V, W) => X)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W], hrx: HailRep[X]) = {
    bind(name, FunType(hrt.typ, hru.typ, hrv.typ, hrw.typ), Arity4Fun[T, U, V, W, X](hrx.typ))
  }

  def registerDependent[T, U, V, W, X](name: String, impl: () => (T, U, V, W) => X)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W], hrx: HailRep[X]) = {
    bind(name, FunType(hrt.typ, hru.typ, hrv.typ, hrw.typ), Arity4DependentFun[T, U, V, W, X](hrx.typ))
  }

  def registerSpecial[T, U, V, W, X, Y, Z](name: String, impl: (() => Any, () => Any, () => Any, () => Any, () => Any, () => Any) => Z)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W], hrx: HailRep[X], hry: HailRep[Y], hrz: HailRep[Z]) = {
    bind(name, FunType(hrt.typ, hru.typ, hrv.typ, hrw.typ, hrx.typ, hry.typ), Arity6Special[T, U, V, W, X, Y, Z](hrz.typ))
  }

  def register[T, U, V, W, X, Y](name: String, impl: (T, U, V, W, X) => Y)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W], hrx: HailRep[X], hry: HailRep[Y]) = {
    bind(name, FunType(hrt.typ, hru.typ, hrv.typ, hrw.typ, hrx.typ), Arity5Fun[T, U, V, W, X, Y](hry.typ))
  }

  def registerDependent[T, U, V, W, X, Y](name: String, impl: () => (T, U, V, W, X) => Y)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W], hrx: HailRep[X], hry: HailRep[Y]) = {
    bind(name, FunType(hrt.typ, hru.typ, hrv.typ, hrw.typ, hrx.typ), Arity5DependentFun[T, U, V, W, X, Y](hry.typ))
  }

  def registerAnn[T](name: String, t: TStruct, impl: T => Annotation)
    (implicit hrt: HailRep[T]) = {
    register(name, impl)(hrt, new HailRep[Annotation] {
      def typ = t
    })
  }

  def registerAnn[T, U](name: String, t: TStruct, impl: (T, U) => Annotation)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    register(name, impl)(hrt, hru, new HailRep[Annotation] {
      def typ = t
    })
  }

  def registerAnn[T, U, V](name: String, t: TStruct, impl: (T, U, V) => Annotation)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    register(name, impl)(hrt, hru, hrv, new HailRep[Annotation] {
      def typ = t
    })
  }

  def registerAnn[T, U, V, W](name: String, t: TStruct, impl: (T, U, V, W) => Annotation)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    register(name, impl)(hrt, hru, hrv, hrw, new HailRep[Annotation] {
      def typ = t
    })
  }

  def registerAnn[T, U, V, W, X](name: String, t: TStruct, impl: (T, U, V, W, X) => Annotation)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W], hrx: HailRep[X]) = {
    register(name, impl)(hrt, hru, hrv, hrw, hrx, new HailRep[Annotation] {
      def typ = t
    })
  }

  def registerAggregator[T, U](name: String, ctor: () => TypedAggregator[U])
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, MethodType(hrt.typ), Arity0Aggregator[T, U](hru.typ))
  }

  def registerDependentAggregator[T, U](name: String, ctor: () => (() => TypedAggregator[U]))
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, MethodType(hrt.typ), Arity0DependentAggregator[T, U](hru.typ))
  }

  def registerLambdaAggregator[T, U, V](name: String, ctor: ((Any) => Any) => TypedAggregator[V])
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, MethodType(hrt.typ, hru.typ), UnaryLambdaAggregator[T, U, V](hrv.typ))
  }

  def registerLambdaAggregator[T, U, V, W](name: String, ctor: ((Any) => Any, V) => TypedAggregator[W])
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    bind(name, MethodType(hrt.typ, hru.typ, hrv.typ), BinaryLambdaAggregator[T, U, V, W](hrw.typ))
  }

  def registerDependentLambdaAggregator[T, U, V, W](name: String, ctor: () => (((Any) => Any, V) => TypedAggregator[W]))
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    bind(name, MethodType(hrt.typ, hru.typ, hrv.typ), BinaryDependentLambdaAggregator[T, U, V, W](hrw.typ))
  }

  def registerAggregator[T, U, V](name: String, ctor: (U) => TypedAggregator[V])
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, MethodType(hrt.typ, hru.typ), Arity1Aggregator[T, U, V](hrv.typ))
  }

  def registerDependentAggregator[T, U, V](name: String, ctor: () => ((U) => TypedAggregator[V]))
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, MethodType(hrt.typ, hru.typ), Arity1DependentAggregator[T, U, V](hrv.typ))
  }

  def registerAggregator[T, U, V, W, X](name: String, ctor: (U, V, W) => TypedAggregator[X])
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W], hrx: HailRep[X]) = {
    bind(name, MethodType(hrt.typ, hru.typ, hrv.typ, hrw.typ), Arity3Aggregator[T, U, V, W, X](hrx.typ))
  }

  val TT = TVariable("T")
  val TU = TVariable("U")
  val TV = TVariable("V")

  val TTBoxed = TVariable("T")

  val TTHr = new HailRep[Any] {
    def typ = TT
  }
  val TUHr = new HailRep[Any] {
    def typ = TU
  }
  val TVHr = new HailRep[Any] {
    def typ = TV
  }
  val BoxedTTHr = new HailRep[AnyRef] {
    def typ = TTBoxed
  }

  val RG = RGVariable()

  private def nonceToNullable[T: TypeInfo, U >: Null](check: Code[T] => Code[Boolean], v: Code[T], ifPresent: Code[T] => Code[U]): CM[Code[U]] = for (
    (stx, x) <- CM.memoize(v)
  ) yield Code(stx, check(x).mux(Code._null[U], ifPresent(x)))

  registerMethod("ploidy", { (c: Call) => Call.ploidy(c) })(callHr, int32Hr)
  registerMethod("isPhased", { (c: Call) => Call.isPhased(c) })(callHr, boolHr)
  // FIXME: Add this when arrays are efficient
  // registerMethod("alleles", { (c: Call) => Call.alleles(c).toFastIndexedSeq })(callHr, arrayHr(int32Hr))
  registerMethod("isHomRef", { (c: Call) => Call.isHomRef(c) })(callHr, boolHr)
  registerMethod("isHet", { (c: Call) => Call.isHet(c) })(callHr, boolHr)
  registerMethod("isHomVar", { (c: Call) => Call.isHomVar(c) })(callHr, boolHr)
  registerMethod("isNonRef", { (c: Call) => Call.isNonRef(c) })(callHr, boolHr)
  registerMethod("isHetNonRef", { (c: Call) => Call.isHetNonRef(c) })(callHr, boolHr)
  registerMethod("isHetRef", { (c: Call) => Call.isHetRef(c) })(callHr, boolHr)
  registerMethod("nNonRefAlleles", { (c: Call) => Call.nNonRefAlleles(c) })(callHr, int32Hr)
  registerMethod("unphasedDiploidGtIndex", { (c: Call) => Call.unphasedDiploidGtIndex(c) })(callHr, int32Hr)
  registerMethod("[]", (c: Call, i: Int) => Call.alleleByIndex(c, i))(callHr, int32Hr, int32Hr)
  registerMethod("oneHotAlleles", { (c: Call, nAlleles: Int) => Call.oneHotAlleles(c, nAlleles) })(callHr, int32Hr, arrayHr(int32Hr))

  register("min_rep", { (l: Locus, a: IndexedSeq[String]) =>
    val (newLocus, newAlleles) = VariantMethods.minRep(l, a)
    Row(newLocus, newAlleles)
  })(
    locusHr(RG), arrayHr(stringHr), new HailRep[Row] { def typ = TTuple(TLocus(RG), TArray(TString())) })
  registerField("contig", { (x: Locus) => x.contig })(locusHr(RG), stringHr)
  registerField("position", { (x: Locus) => x.position })(locusHr(RG), int32Hr)
  registerField("start", { (x: Interval) => x.start })(intervalHr(TTHr), TTHr)
  registerField("end", { (x: Interval) => x.end })(intervalHr(TTHr), TTHr)
  registerField("includesStart", { (x: Interval) => x.includesStart })(intervalHr(TTHr), boolHr)
  registerField("includesEnd", { (x: Interval) => x.includesEnd })(intervalHr(TTHr), boolHr)

  register("is_snp", { (ref: String, alt: String) => AltAlleleMethods.isSNP(ref, alt) })
  register("is_mnp", { (ref: String, alt: String) => AltAlleleMethods.isMNP(ref, alt) })
  register("is_transition", { (ref: String, alt: String) => AltAlleleMethods.isTransition(ref, alt) })
  register("is_transversion", { (ref: String, alt: String) => AltAlleleMethods.isTransversion(ref, alt) })
  register("is_insertion", { (ref: String, alt: String) => AltAlleleMethods.isInsertion(ref, alt) })
  register("is_deletion", { (ref: String, alt: String) => AltAlleleMethods.isDeletion(ref, alt) })
  register("is_indel", { (ref: String, alt: String) => AltAlleleMethods.isIndel(ref, alt) })
  register("is_star", { (ref: String, alt: String) => AltAlleleMethods.isStar(ref, alt) })
  register("is_complex", { (ref: String, alt: String) => AltAlleleMethods.isComplex(ref, alt) })
  register("allele_type", { (ref: String, alt: String) => AltAlleleMethods.altAlleleType(ref, alt).toString })
  register("hamming", { (ref: String, alt: String) => AltAlleleMethods.hamming(ref, alt) })
  registerMethodDependent("inXPar", { () =>
    val rg = RG.rg
    (x: Locus) => x.inXPar(rg)
  })(locusHr(RG), boolHr)
  registerMethodDependent("inYPar", { () =>
    val rg = RG.rg
    (x: Locus) => x.inYPar(rg)
  })(locusHr(RG), boolHr)
  registerMethodDependent("inXNonPar", { () =>
    val rg = RG.rg
    (x: Locus) => x.inXNonPar(rg)
  })(locusHr(RG), boolHr)
  registerMethodDependent("inYNonPar", { () =>
    val rg = RG.rg
    (x: Locus) => x.inYNonPar(rg)
  })(locusHr(RG), boolHr)
  registerMethodDependent("isAutosomal", { () =>
    val rg = RG.rg
    (x: Locus) => x.isAutosomal(rg)
  })(locusHr(RG), boolHr)
  registerMethodDependent("isAutosomalOrPseudoAutosomal", { () =>
    val rg = RG.rg
    (x: Locus) => x.isAutosomalOrPseudoAutosomal(rg)
  })(locusHr(RG), boolHr)
  registerMethodDependent("isMitochondrial", { () =>
    val rg = RG.rg
    (x: Locus) => x.isMitochondrial(rg)
  })(locusHr(RG), boolHr)

  register("triangle", { (i: Int) => triangle(i) })

  register("plDosage", { (pl: IndexedSeq[Int]) =>
    if (pl.length != 3)
      fatal(s"length of pl array must be 3, got ${ pl.length }")
    Genotype.plToDosage(pl(0), pl(1), pl(2))
  })

  register("dosage", { (gp: IndexedSeq[Double]) =>
    if (gp.length != 3)
      fatal(s"length of gp array must be 3, got ${ gp.length }")
    gp(1) + 2.0 * gp(2)
  })

  register("downcode", { (c: Call, i: Int) =>
    Call.downcode(c, i)
  })(callHr, int32Hr, callHr)

  register("gqFromPL", { pl: IndexedSeq[Int] =>
    // FIXME toArray
    Genotype.gqFromPL(pl.toArray)
  })(arrayHr(int32Hr), int32Hr)

  registerMethod("length", { (x: String) => x.length })

  registerUnaryNAFilteredCollectionMethod("sum", { (x: TraversableOnce[Int]) => x.sum })
  registerUnaryNAFilteredCollectionMethod("sum", { (x: TraversableOnce[Long]) => x.sum })
  registerUnaryNAFilteredCollectionMethod("sum", { (x: TraversableOnce[Float]) => x.sum })
  registerUnaryNAFilteredCollectionMethod("sum", { (x: TraversableOnce[Double]) => x.sum })

  registerUnaryNAFilteredCollectionMethod("product", { (x: TraversableOnce[Int]) => x.product })
  registerUnaryNAFilteredCollectionMethod("product", { (x: TraversableOnce[Long]) => x.product })
  registerUnaryNAFilteredCollectionMethod("product", { (x: TraversableOnce[Float]) => x.product })
  registerUnaryNAFilteredCollectionMethod("product", { (x: TraversableOnce[Double]) => x.product })

  registerUnaryNAFilteredCollectionMethod("min", { (x: TraversableOnce[Int]) => if (x.nonEmpty) box(x.min) else null })(int32Hr, boxedInt32Hr)
  registerUnaryNAFilteredCollectionMethod("min", { (x: TraversableOnce[Long]) => if (x.nonEmpty) box(x.min) else null })(int64Hr, boxedInt64Hr)
  registerUnaryNAFilteredCollectionMethod("min", { (x: TraversableOnce[Float]) => if (x.nonEmpty) box(x.min) else null })(float32Hr, boxedFloat32Hr)
  registerUnaryNAFilteredCollectionMethod("min", { (x: TraversableOnce[Double]) => if (x.nonEmpty) box(x.min) else null })(float64Hr, boxedFloat64Hr)

  registerUnaryNAFilteredCollectionMethod("max", { (x: TraversableOnce[Int]) => if (x.nonEmpty) box(x.max) else null })(int32Hr, boxedInt32Hr)
  registerUnaryNAFilteredCollectionMethod("max", { (x: TraversableOnce[Long]) => if (x.nonEmpty) box(x.max) else null })(int64Hr, boxedInt64Hr)
  registerUnaryNAFilteredCollectionMethod("max", { (x: TraversableOnce[Float]) => if (x.nonEmpty) box(x.max) else null })(float32Hr, boxedFloat32Hr)
  registerUnaryNAFilteredCollectionMethod("max", { (x: TraversableOnce[Double]) => if (x.nonEmpty) box(x.max) else null })(float64Hr, boxedFloat64Hr)

  registerUnaryNAFilteredCollectionMethod("median", { (x: TraversableOnce[Int]) => if (x.nonEmpty) box(breeze.stats.median(DenseVector(x.toArray))) else null })(int32Hr, boxedInt32Hr)
  registerUnaryNAFilteredCollectionMethod("median", { (x: TraversableOnce[Long]) => if (x.nonEmpty) box(breeze.stats.median(DenseVector(x.toArray))) else null })(int64Hr, boxedInt64Hr)
  registerUnaryNAFilteredCollectionMethod("median", { (x: TraversableOnce[Float]) => if (x.nonEmpty) box(breeze.stats.median(DenseVector(x.toArray))) else null })(float32Hr, boxedFloat32Hr)
  registerUnaryNAFilteredCollectionMethod("median", { (x: TraversableOnce[Double]) => if (x.nonEmpty) box(breeze.stats.median(DenseVector(x.toArray))) else null })(float64Hr, boxedFloat64Hr)

  registerUnaryNAFilteredCollectionMethod("mean", { (x: TraversableOnce[Int]) => if (x.nonEmpty) box(x.sum / x.size.toDouble) else null })(int32Hr, boxedFloat64Hr)
  registerUnaryNAFilteredCollectionMethod("mean", { (x: TraversableOnce[Long]) => if (x.nonEmpty) box(x.sum / x.size.toDouble) else null })(int64Hr, boxedFloat64Hr)
  registerUnaryNAFilteredCollectionMethod("mean", { (x: TraversableOnce[Float]) => if (x.nonEmpty) box(x.sum / x.size.toDouble) else null })(float32Hr, boxedFloat64Hr)
  registerUnaryNAFilteredCollectionMethod("mean", { (x: TraversableOnce[Double]) => if (x.nonEmpty) box(x.sum / x.size.toDouble) else null })(float64Hr, boxedFloat64Hr)

  register("range", { (x: Int) =>
    val a = if (x <= 0) {
      new Array[Int](0)
    } else {
      val a = new Array[Int](x)
      var i = 0
      while (i < x) {
        a(i) = i
        i += 1
      }
      a
    }
    a: IndexedSeq[Int]
  })
  register("range", { (x: Int, y: Int) =>
    val a = if (y <= x) {
      new Array[Int](0)
    } else {
      val l = y - x
      val a = new Array[Int](l)
      var i = 0
      while (i < l) {
        a(i) = x + i
        i += 1
      }
      a
    }
    a: IndexedSeq[Int]
  })
  register("range", { (x: Int, y: Int, step: Int) => (x until y by step).toArray: IndexedSeq[Int] })

  register("Call", { (phased: Boolean) => Call0(phased) })(boolHr, callHr)
  register("Call", { (i: Int, phased: Boolean) => Call1(i, phased) })(int32Hr, boolHr, callHr)
  register("Call", { (i: Int, j: Int, phased: Boolean) => Call2(i, j, phased) })(int32Hr, int32Hr, boolHr, callHr)
  register("Call", { (alleles: IndexedSeq[Int], phased: Boolean) => CallN(alleles.toArray, phased) })(arrayHr(int32Hr), boolHr, callHr)
  register("Call", { (s: String) => Call.parse(s) })(stringHr, callHr)
  register("UnphasedDiploidGtIndexCall", { (gt: Int) => Call2.fromUnphasedDiploidGtIndex(gt) })(int32Hr, callHr)

  val tuple2Hr = new HailRep[Annotation] {
    override def typ: Type = TTuple(TTHr.typ, TUHr.typ)
  }
  register("dict", { (elements: IndexedSeq[Annotation]) =>
    elements.filter(_ != null).map { e =>
      val r = e.asInstanceOf[Row]
      (r.get(0), r.get(1))
  }.toMap
  })(arrayHr(tuple2Hr), dictHr(TTHr, TUHr))

  register("dict", { (elements: Set[Annotation]) =>
    elements.filter(_ != null).map { e =>
      val r = e.asInstanceOf[Row]
      (r.get(0), r.get(1))
    }.toMap
  })(setHr(tuple2Hr), dictHr(TTHr, TUHr))

  register("dictToArray", (m: Map[Annotation, Annotation]) => {
    m.map { case (k, v) => Annotation(k, v) }.toArray.toFastIndexedSeq
  })(dictHr(TTHr, TUHr), arrayHr(tuple2Hr))

  registerDependent("Locus", { () =>
    val rg = RG.rg
    (x: String) => Locus.parse(x, rg)
  })(stringHr, locusHr(RG))

  val locusAllelesHr = new HailRep[Annotation] {
    def typ = TStruct("locus" -> TLocus(RG), "alleles" -> TArray(TString()))
  }
  registerDependent("LocusAlleles", { () =>
    val rg = RG.rg
    (s: String) => {
      val (locus, alleles) = VariantMethods.parse(s, rg)
      Annotation(locus, alleles)
    }
  })(stringHr, locusAllelesHr)

  registerDependent("Locus", { () =>
    val rg = RG.rg
    (contig: String, pos: Int) => Locus(contig, pos, rg)
    })(stringHr, int32Hr, locusHr(RG))

  registerDependent("Interval", () => {
    val t = TT.t
    (x: Annotation, y: Annotation, includesStart: Boolean, includesEnd: Boolean) => Interval(x, y, includesStart, includesEnd)
  })(TTHr, TTHr, boolHr, boolHr, intervalHr(TTHr))

  registerAnn("hwe", hweStruct, { (nHomRef: Int, nHet: Int, nHomVar: Int) =>
    val res = hweTest(nHomRef, nHet, nHomVar)
    Annotation(res(0), res(1))
  })

  registerAnn("chisq", chisqStruct, { (a: Int, b: Int, c: Int, d: Int) =>
    val res = chisqTest(a, b, c, d)
    Annotation(res(0), res(1))
  })

  registerAnn("ctt", chisqStruct, { (a: Int, b: Int, c: Int, d: Int, minCellCount: Int) =>
    val ctt = contingencyTableTest(a, b, c, d, minCellCount)
    Annotation(ctt(0), ctt(1))
  })

  registerAnn("fet", fetStruct, { (a: Int, b: Int, c: Int, d: Int) =>
    val fet = fisherExactTest(a, b, c, d)
    Annotation(fet(0), fet(1), fet(2), fet(3))
  })

  register("binomTest", { (x: Int, n: Int, p: Double, alternative: Int) => binomTest(x, n, p, alternative)
  })

  // NB: merge takes two structs, how do I deal with structs?
  register("exp", { (x: Double) => math.exp(x) })
  register("log10", { (x: Double) => math.log10(x) })
  register("sqrt", { (x: Double) => math.sqrt(x) })
  register("log", (x: Double) => math.log(x))
  register("log", (x: Double, b: Double) => math.log(x) / math.log(b))

  register("gamma", (x: Double) => Gamma.gamma(x))

  registerDependent("LocusInterval", () => {
    val rg = RG.rg
   (s: String) => Locus.parseInterval(s, rg)
  })(stringHr, intervalHr(locusHr(RG)))

  registerDependent("LocusInterval", () => {
    val rg = RG.rg
    (chr: String, start: Int, end: Int, includesStart: Boolean, includesEnd: Boolean) => Locus.makeInterval(chr, start, end, includesStart, includesEnd, rg)
  })(stringHr, int32Hr, int32Hr, boolHr, boolHr, intervalHr(locusHr(RG)))

  register("pcoin", { (p: Double) => math.random < p })
  register("runif", { (min: Double, max: Double) => min + (max - min) * math.random })

  register("dbeta", { (x: Double, a: Double, b: Double) => dbeta(x, a, b) })
  register("rnorm", { (mean: Double, sd: Double) => mean + sd * scala.util.Random.nextGaussian() })
  register("pnorm", { (x: Double) => pnorm(x) })
  register("qnorm", { (p: Double) => qnorm(p) })

  register("rpois", { (lambda: Double) => rpois(lambda) })
  register("rpois", { (n: Int, lambda: Double) => rpois(n, lambda).toFastIndexedSeq })(int32Hr, float64Hr, arrayHr(float64Hr))
  register("dpois", { (x: Double, lambda: Double) => dpois(x, lambda) })
  register("dpois", { (x: Double, lambda: Double, logP: Boolean) => dpois(x, lambda, logP) })
  register("ppois", { (x: Double, lambda: Double) => ppois(x, lambda) })
  register("ppois", { (x: Double, lambda: Double, lowerTail: Boolean, logP: Boolean) => ppois(x, lambda, lowerTail, logP) })

  register("qpois", { (p: Double, lambda: Double) => qpois(p, lambda) })

  register("qpois", { (p: Double, lambda: Double, lowerTail: Boolean, logP: Boolean) => qpois(p, lambda, lowerTail, logP) })

  register("pchisqtail", { (x: Double, df: Double) => chiSquaredTail(x, df) })
  register("qchisqtail", { (p: Double, df: Double) => inverseChiSquaredTail(p, df) })

  register("!", (a: Boolean) => !a)

  def iToD(x: Code[java.lang.Integer]): Code[java.lang.Double] =
    Code.boxDouble(Code.intValue(x).toD)

  def iToF(x: Code[java.lang.Integer]): Code[java.lang.Float] =
    Code.boxFloat(Code.floatValue(x).toF)

  def lToD(x: Code[java.lang.Long]): Code[java.lang.Double] =
    Code.boxDouble(Code.longValue(x).toD)

  def lToF(x: Code[java.lang.Long]): Code[java.lang.Float] =
    Code.boxFloat(Code.longValue(x).toF)

  def iToL(x: Code[java.lang.Integer]): Code[java.lang.Long] =
    Code.boxLong(Code.intValue(x).toL)

  def fToD(x: Code[java.lang.Float]): Code[java.lang.Double] =
    Code.boxDouble(Code.floatValue(x).toD)

  registerMethod("split", (s: String, p: String) => s.split(p): IndexedSeq[String])

  registerMethod("split", (s: String, p: String, n: Int) => s.split(p, n): IndexedSeq[String])

  registerMethod("lower", (s: String) => s.toLowerCase)
  registerMethod("upper", (s: String) => s.toUpperCase)
  registerMethod("strip", (s: String) => s.trim())
  registerMethod("contains", (s: String, t: String) => s.contains(t))
  registerMethod("startswith", (s: String, t: String) => s.startsWith(t))
  registerMethod("endswith", (s: String, t: String) => s.endsWith(t))
  registerMethod("firstMatchIn", (s: String, regex: String) => regex.r.findFirstMatchIn(s).map(_.subgroups.toArray.toFastIndexedSeq).orNull : IndexedSeq[String])

  registerMethod("replace", (str: String, pattern1: String, pattern2: String) =>
    str.replaceAll(pattern1, pattern2))

  registerMethod("entropy", { (x: String) => entropy(x)
  })

  registerMethodDependent("contains", () => {
    val pord = TT.t.ordering
    (interval: Interval, point: Annotation) => interval.contains(pord, point)
  })(intervalHr(TTHr), TTHr, boolHr)

  registerMethodDependent("overlaps", () => {
    val pord = TT.t.ordering
    (interval1: Interval, interval2: Interval) => interval1.mayOverlap(pord, interval2)
  })(intervalHr(TTHr), intervalHr(TTHr), boolHr)

  registerMethod("length", (a: IndexedSeq[Any]) => a.length)(arrayHr(TTHr), int32Hr)
  registerMethod("size", (a: IndexedSeq[Any]) => a.size)(arrayHr(TTHr), int32Hr)
  registerMethod("size", (s: Set[Any]) => s.size)(setHr(TTHr), int32Hr)
  registerMethod("size", (d: Map[Any, Any]) => d.size)(dictHr(TTHr, TUHr), int32Hr)
  registerMethod("size", (s: String) => s.length)(stringHr, int32Hr)

  registerMethod("isEmpty", (a: IndexedSeq[Any]) => a.isEmpty)(arrayHr(TTHr), boolHr)
  registerMethod("isEmpty", (s: Set[Any]) => s.isEmpty)(setHr(TTHr), boolHr)
  registerMethod("isEmpty", (d: Map[Any, Any]) => d.isEmpty)(dictHr(TTHr, TUHr), boolHr)

  registerMethod("toSet", (a: IndexedSeq[Any]) => a.toSet)(arrayHr(TTHr), setHr(TTHr))
  registerMethod("toSet", (a: Set[Any]) => a)(setHr(TTHr), setHr(TTHr))
  registerMethod("toArray", (a: Set[Any]) => a.toArray[Any]: IndexedSeq[Any])(setHr(TTHr), arrayHr(TTHr))
  registerMethod("toArray", (a: IndexedSeq[Any]) => a)(arrayHr(TTHr), arrayHr(TTHr))

  registerMethod("append", (x: IndexedSeq[Any], a: Any) => x :+ a)(arrayHr(TTHr), TTHr, arrayHr(TTHr))
  registerMethod("extend", (x: IndexedSeq[Any], a: IndexedSeq[Any]) => x ++ a)(arrayHr(TTHr), arrayHr(TTHr), arrayHr(TTHr))

  registerMethod("add", (x: Set[Any], a: Any) => x + a)(setHr(TTHr), TTHr, setHr(TTHr))
  registerMethod("remove", (x: Set[Any], a: Any) => x - a)(setHr(TTHr), TTHr, setHr(TTHr))
  registerMethod("union", (x: Set[Any], a: Set[Any]) => x ++ a)(setHr(TTHr), setHr(TTHr), setHr(TTHr))
  registerMethod("intersection", (x: Set[Any], a: Set[Any]) => x & a)(setHr(TTHr), setHr(TTHr), setHr(TTHr))
  registerMethod("difference", (x: Set[Any], a: Set[Any]) => x &~ a)(setHr(TTHr), setHr(TTHr), setHr(TTHr))
  registerMethod("isSubset", (x: Set[Any], a: Set[Any]) => x.subsetOf(a))(setHr(TTHr), setHr(TTHr), boolHr)

  registerMethod("flatten", (a: IndexedSeq[IndexedSeq[Any]]) =>
    flattenOrNull[IndexedSeq](IndexedSeq.newBuilder[Any], a))(arrayHr(arrayHr(TTHr)), arrayHr(TTHr))

  registerMethod("flatten", (s: Set[Set[Any]]) =>
    flattenOrNull[Set](Set.newBuilder[Any], s))(setHr(setHr(TTHr)), setHr(TTHr))

  registerMethod("keys", (m: Map[Any, Any]) =>
    m.keysIterator.toArray[Any]: IndexedSeq[Any])(dictHr(TTHr, TUHr), arrayHr(TTHr))

  registerMethod("values", (m: Map[Any, Any]) =>
    m.valuesIterator.toArray[Any]: IndexedSeq[Any])(dictHr(TTHr, TUHr), arrayHr(TUHr))

  registerMethod("keySet", (m: Map[Any, Any]) =>
    m.keySet)(dictHr(TTHr, TUHr), setHr(TTHr))

  registerMethod("get", (m: Map[Any, Any], key: Any) =>
    m.get(key).orNull)(dictHr(TTHr, TUHr), TTHr, TUHr)

  registerMethod("get", (m: Map[Any, Any], key: Any, default: Any) =>
    m.getOrElse(key, default))(dictHr(TTHr, TUHr), TTHr, TUHr, TUHr)

  registerMethod("mkString", (a: IndexedSeq[String], d: String) => a.mkString(d)
  )(arrayHr(stringHr), stringHr, stringHr)
  registerMethod("mkString", (s: Set[String], d: String) => s.mkString(d)
  )(setHr(stringHr), stringHr, stringHr)

  registerMethod("contains", (s: Set[Any], x: Any) => s.contains(x)
  )(setHr(TTHr), TTHr, boolHr)
  registerMethod("contains", (d: Map[Any, Any], x: Any) => d.contains(x)
  )(dictHr(TTHr, TUHr), TTHr, boolHr)

  registerLambda("uniroot", { (f: (Any) => Any, min: Double, max: Double) =>
    val r = uniroot({ (x: Double) =>
      if (!(min < max))
        fatal(s"min must be less than max in call to uniroot, got: min $min, max $max")

      val fmin = f(min)
      val fmax = f(max)

      if (fmin == null || fmax == null)
        fatal(s"result of f($x) missing in call to uniroot")

      if (fmin.asInstanceOf[Double] * fmax.asInstanceOf[Double] > 0.0)
        fatal(s"sign of endpoints must have opposite signs, got: f(min) = $fmin, f(max) = $fmax")

      val y = f(x)
      if (y == null)
        fatal(s"result of f($x) missing in call to uniroot")
      y.asInstanceOf[Double]
    }, min, max)
    (r match {
      case Some(r) => r
      case None => null
    }): java.lang.Double
  }
  )(unaryHr(float64Hr, float64Hr), float64Hr, float64Hr, boxedFloat64Hr)

  registerLambdaMethod("map", (a: IndexedSeq[Any], f: (Any) => Any) => {
    val l = a.length
    val r = new Array[Any](a.length)
    var i = 0
    while (i < l) {
      r(i) = f(a(i))
      i += 1
    }

    r: IndexedSeq[Any]
  })(arrayHr(TTHr), unaryHr(TTHr, TUHr), arrayHr(TUHr))

  registerLambdaMethod("map", (s: Set[Any], f: (Any) => Any) =>
    s.map(f)
  )(setHr(TTHr), unaryHr(TTHr, TUHr), setHr(TUHr))

  registerLambdaMethod("mapValues", (a: Map[Any, Any], f: (Any) => Any) =>
    a.map { case (k, v) => (k, f(v)) }
  )(dictHr[Any, Any](TTHr, TUHr), unaryHr(TUHr, TVHr), dictHr[Any, Any](TTHr, TVHr))

  //  registerMapLambdaMethod("mapValues", (a: Map[Any, Any], f: (Any) => Any) =>
  //    a.map { case (k, v) => (k, f(v)) }
  //  )(TTHr, TUHr, unaryHr(TUHr, TVHr), TVHr)

  registerLambdaMethod("flatMap", (a: IndexedSeq[Any], f: (Any) => Any) =>
    flattenOrNull[IndexedSeq](IndexedSeq.newBuilder[Any],
      a.map(f).asInstanceOf[IndexedSeq[IndexedSeq[Any]]])
  )(arrayHr(TTHr), unaryHr(TTHr, arrayHr(TUHr)), arrayHr(TUHr))

  registerLambdaMethod("flatMap", (s: Set[Any], f: (Any) => Any) =>
    flattenOrNull[Set](Set.newBuilder[Any],
      s.map(f).asInstanceOf[Set[Set[Any]]])
  )(setHr(TTHr), unaryHr(TTHr, setHr(TUHr)), setHr(TUHr))

  registerLambdaMethod("groupBy", (a: IndexedSeq[Any], f: (Any) => Any) =>
    a.groupBy(f)
  )(arrayHr(TTHr), unaryHr(TTHr, TUHr), dictHr(TUHr, arrayHr(TTHr)))

  registerLambdaMethod("groupBy", (a: Set[Any], f: (Any) => Any) =>
    a.groupBy(f)
  )(setHr(TTHr), unaryHr(TTHr, TUHr), dictHr(TUHr, setHr(TTHr)))

  registerLambdaMethod("exists", (a: IndexedSeq[Any], f: (Any) => Any) =>
    a.exists { x =>
      val r = f(x)
      r != null && r.asInstanceOf[Boolean]
    })(arrayHr(TTHr), unaryHr(TTHr, boolHr), boolHr)

  registerLambdaMethod("exists", (s: Set[Any], f: (Any) => Any) =>
    s.exists { x =>
      val r = f(x)
      r != null && r.asInstanceOf[Boolean]
    })(setHr(TTHr), unaryHr(TTHr, boolHr), boolHr)

  registerLambdaMethod("forall", (a: IndexedSeq[Any], f: (Any) => Any) =>
    a.forall { x =>
      val r = f(x)
      r != null && r.asInstanceOf[Boolean]
    })(arrayHr(TTHr), unaryHr(TTHr, boolHr), boolHr)

  registerLambdaMethod("forall", (s: Set[Any], f: (Any) => Any) =>
    s.forall { x =>
      val r = f(x)
      r != null && r.asInstanceOf[Boolean]
    })(setHr(TTHr), unaryHr(TTHr, boolHr), boolHr)

  registerLambdaMethod("filter", (a: IndexedSeq[Any], f: (Any) => Any) => {
    val b = new ArrayBuilder[Any](
      math.min(a.length, ArrayBuilder.defaultInitialCapacity))
    val l = a.length
    var i = 0
    while (i < l) {
      val x = a(i)
      val p = f(x)
      if (p != null && p.asInstanceOf[Boolean]) {
        b += x
      }
      i += 1
    }
    new TruncatedArrayIndexedSeq(b.underlying(), b.length): IndexedSeq[Any]
  })(arrayHr(TTHr), unaryHr(TTHr, boolHr), arrayHr(TTHr))

  registerLambdaMethod("filter", (s: Set[Any], f: (Any) => Any) =>
    s.filter { x =>
      val r = f(x)
      r != null && r.asInstanceOf[Boolean]
    })(setHr(TTHr), unaryHr(TTHr, boolHr), setHr(TTHr))

  registerAggregator[Any, Long]("count", () => new CountAggregator()
  )(aggregableHr(TTHr), int64Hr)

  registerDependentAggregator[Any, IndexedSeq[Any]]("collect", () => {
    val t = TT.t
    () => new CollectAggregator(t)
  })(aggregableHr(TTHr), arrayHr(TTHr))

  registerDependentAggregator[Any, Set[Any]]("collectAsSet", () => {
    val t = TT.t
    () => new CollectSetAggregator(t)
  })(aggregableHr(TTHr), setHr(TTHr))

  registerAggregator[Int, Int]("sum", () => new SumAggregator[Int]())(aggregableHr(int32Hr), int32Hr)

  registerAggregator[Long, Long]("sum", () => new SumAggregator[Long]())(aggregableHr(int64Hr), int64Hr)

  registerAggregator[Float, Float]("sum", () => new SumAggregator[Float]())(aggregableHr(float32Hr), float32Hr)

  registerAggregator[Double, Double]("sum", () => new SumAggregator[Double]())(aggregableHr(float64Hr), float64Hr)

  registerAggregator[IndexedSeq[Int], IndexedSeq[Int]]("sum", () => new SumArrayAggregator[Int]()
  )(aggregableHr(arrayHr(int32Hr)), arrayHr(int32Hr))

  registerAggregator[IndexedSeq[Long], IndexedSeq[Long]]("sum", () => new SumArrayAggregator[Long]()
  )(aggregableHr(arrayHr(int64Hr)), arrayHr(int64Hr))

  registerAggregator[IndexedSeq[Float], IndexedSeq[Float]]("sum", () => new SumArrayAggregator[Float]()
  )(aggregableHr(arrayHr(float32Hr)), arrayHr(float32Hr))

  registerAggregator[IndexedSeq[Double], IndexedSeq[Double]]("sum", () => new SumArrayAggregator[Double]()
  )(aggregableHr(arrayHr(float64Hr)), arrayHr(float64Hr))

  registerAggregator[Long, Long]("product", () => new ProductAggregator[Long]())(aggregableHr(int64Hr), int64Hr)

  registerAggregator[Double, Double]("product", () => new ProductAggregator[Double]())(aggregableHr(float64Hr), float64Hr)

  registerAggregator[Int, java.lang.Integer]("max", () => new MaxAggregator[Int, java.lang.Integer]())(aggregableHr(int32Hr), boxedInt32Hr)

  registerAggregator[Long, java.lang.Long]("max", () => new MaxAggregator[Long, java.lang.Long]())(aggregableHr(int64Hr), boxedInt64Hr)

  registerAggregator[Float, java.lang.Float]("max", () => new MaxAggregator[Float, java.lang.Float]())(aggregableHr(float32Hr), boxedFloat32Hr)

  registerAggregator[Double, java.lang.Double]("max", () => new MaxAggregator[Double, java.lang.Double]())(aggregableHr(float64Hr), boxedFloat64Hr)

  registerAggregator[Int, java.lang.Integer]("min", () => new MinAggregator[Int, java.lang.Integer]())(aggregableHr(int32Hr), boxedInt32Hr)

  registerAggregator[Long, java.lang.Long]("min", () => new MinAggregator[Long, java.lang.Long]())(aggregableHr(int64Hr), boxedInt64Hr)

  registerAggregator[Float, java.lang.Float]("min", () => new MinAggregator[Float, java.lang.Float]())(aggregableHr(float32Hr), boxedFloat32Hr)

  registerAggregator[Double, java.lang.Double]("min", () => new MinAggregator[Double, java.lang.Double]())(aggregableHr(float64Hr), boxedFloat64Hr)

  registerAggregator[IndexedSeq[java.lang.Double], Any]("infoScore", () => new InfoScoreAggregator())(aggregableHr(arrayHr(boxedFloat64Hr)),
    new HailRep[Any] {
      def typ: Type = InfoScoreCombiner.signature
    })

  registerAggregator[Call, Any]("hardyWeinberg", () => new HWEAggregator())(aggregableHr(callHr),
    new HailRep[Any] {
      def typ = HWECombiner.signature
    })

  registerDependentAggregator[Any, Any]("counter", () => {
    val t = TT.t
    () => new CounterAggregator(t)
  })(aggregableHr(TTHr),
    new HailRep[Any] {
      def typ = TDict(TTHr.typ, TInt64())
    })

  registerAggregator[Double, Any]("stats", () => new StatAggregator())(aggregableHr(float64Hr),
    new HailRep[Any] {
      def typ = TStruct("mean" -> TFloat64(), "stdev" -> TFloat64(), "min" -> TFloat64(),
        "max" -> TFloat64(), "n" -> TInt64(), "sum" -> TFloat64())
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
  })(aggregableHr(float64Hr), float64Hr, float64Hr, int32Hr, new HailRep[Any] {
    def typ = HistogramCombiner.schema
  })

  registerLambdaAggregator[Call, (Any) => Any, Any]("callStats", (vf: (Any) => Any) => new CallStatsAggregator(vf)
  )(aggregableHr(callHr), unaryHr(callHr, int32Hr), new HailRep[Any] {
      def typ = CallStats.schema
    })

  registerLambdaAggregator[Call, (Any) => Any, Any]("inbreeding", (af: (Any) => Any) => new InbreedingAggregator(af)
  )(aggregableHr(callHr), unaryHr(callHr, float64Hr), new HailRep[Any] {
      def typ = InbreedingCombiner.signature
    })

  registerLambdaAggregator[Any, (Any) => Any, java.lang.Double]("fraction", (f: (Any) => Any) => new FractionAggregator(f)
  )(aggregableHr(TTHr), unaryHr(TTHr, boxedboolHr), boxedFloat64Hr)

  registerLambdaAggregator[Any, (Any) => Any, Boolean]("exists", (f: (Any) => Any) => new ExistsAggregator(f)
  )(aggregableHr(TTHr), unaryHr(TTHr, boxedboolHr), boolHr)

  registerLambdaAggregator[Any, (Any) => Any, Boolean]("forall", (f: (Any) => Any) => new ForallAggregator(f)
  )(aggregableHr(TTHr), unaryHr(TTHr, boxedboolHr), boolHr)

  registerDependentAggregator("take", () => {
    val t = TT.t
    (n: Int) => new TakeAggregator(t, n)
  })(aggregableHr(TTHr), int32Hr, arrayHr(TTHr))

  registerDependentLambdaAggregator("takeBy", () => {
    val t = TT.t
    (f: (Any) => Any, n: Int) => new TakeByAggregator[Int](t, f, n)
  })(aggregableHr(TTHr), unaryHr(TTHr, boxedInt32Hr), int32Hr, arrayHr(TTHr))

  registerDependentLambdaAggregator("takeBy", () => {
    val t = TT.t
    (f: (Any) => Any, n: Int) => new TakeByAggregator[Long](t, f, n)
  })(aggregableHr(TTHr), unaryHr(TTHr, boxedInt64Hr), int32Hr, arrayHr(TTHr))

  registerDependentLambdaAggregator("takeBy", () => {
    val t = TT.t
    (f: (Any) => Any, n: Int) => new TakeByAggregator[Float](t, f, n)
  })(aggregableHr(TTHr), unaryHr(TTHr, boxedFloat32Hr), int32Hr, arrayHr(TTHr))

  registerDependentLambdaAggregator("takeBy", () => {
    val t = TT.t
    (f: (Any) => Any, n: Int) => new TakeByAggregator[Double](t, f, n)
  })(aggregableHr(TTHr), unaryHr(TTHr, boxedFloat64Hr), int32Hr, arrayHr(TTHr))

  registerDependentLambdaAggregator("takeBy", () => {
    val t = TT.t
    (f: (Any) => Any, n: Int) => new TakeByAggregator[String](t, f, n)
  })(aggregableHr(TTHr), unaryHr(TTHr, stringHr), int32Hr, arrayHr(TTHr))

  val aggST = Box[SymbolTable]()

  registerLambdaAggregatorTransformer("flatMap", { (a: CPS[Any], f: (Any) => Any) => { (k: Any => Unit) =>
    a { x =>
      val r = f(x).asInstanceOf[IndexedSeq[Any]]
      if (r != null) {
        var i = 0
        while (i < r.size) {
          k(r(i))
          i += 1
        }
      }
    }
  }
  }, { (x: Code[AnyRef], f: Code[AnyRef] => CM[Code[AnyRef]]) => { (k: Code[AnyRef] => CM[Code[Unit]]) =>
    for {
      is <- f(x)
      (str, r) <- CM.memoize(Code.checkcast[IndexedSeq[AnyRef]](is))
      (stn, n) <- CM.memoize(r.invoke[Int]("size"))
      i <- CM.newLocal[Int]
      ri = r.invoke[Int, AnyRef]("apply", i)
      invokek <- k(ri)
    } yield Code(
      str,
      r.ifNull(
        Code._empty,
        Code(
          stn,
          i.store(0),
          Code.whileLoop(i < n,
            Code(invokek, i.store(i + 1))
          )))
    )
  }
  })(aggregableHr(TTHr, aggST), unaryHr(TTHr, arrayHr(TUHr)), aggregableHr(TUHr, aggST))

  registerLambdaAggregatorTransformer("flatMap", { (a: CPS[Any], f: (Any) => Any) => { (k: Any => Any) => a { x =>
    val s = f(x).asInstanceOf[Set[Any]]
    if (s != null)
      s.foreach(k) } }
  }, { (x: Code[AnyRef], f: Code[AnyRef] => CM[Code[AnyRef]]) => { (k: Code[AnyRef] => CM[Code[Unit]]) =>
    for {
      fx <- f(x)
      (sts, s) <- CM.memoize(Code.checkcast[Set[AnyRef]](fx))
      (stit, it) <- CM.memoize(s.invoke[Iterator[AnyRef]]("iterator"))
      hasNext = it.invoke[Boolean]("hasNext")
      next = it.invoke[AnyRef]("next")
      invokek <- k(next)
    } yield Code(
      sts,
      s.ifNull(
        Code._empty,
        Code(
          stit,
          Code.whileLoop(hasNext, invokek))))
  }
  })(aggregableHr(TTHr, aggST), unaryHr(TTHr, setHr(TUHr)), aggregableHr(TUHr, aggST))

  registerLambdaAggregatorTransformer("filter", { (a: CPS[Any], f: (Any) => Any) => { (k: Any => Any) =>
    a { x =>
      val r = f(x)
      if (r != null && r.asInstanceOf[Boolean])
        k(x)
    }
  }
  }, { (_x: Code[AnyRef], f: Code[AnyRef] => CM[Code[AnyRef]]) => { (k: Code[AnyRef] => CM[Code[Unit]]) =>
    for (
      (stx, x) <- CM.memoize(_x);
      (str, r) <- CM.memoize(f(x));
      invokek <- k(x)
    ) yield Code(stx, str,
      // NB: the invocation of `k` doesn't modify the stack.
      r.ifNull(Code._empty[Unit],
        Code.booleanValue(Code.checkcast[java.lang.Boolean](r)).mux(
          invokek,
          Code._empty[Unit]))
    )
  }
  })(aggregableHr(TTHr, aggST), unaryHr(TTHr, boolHr), aggregableHr(TTHr, aggST))

  registerLambdaAggregatorTransformer("map", { (a: CPS[Any], f: (Any) => Any) => { (k: Any => Any) => a { x => k(f(x)) } }
  }, { (x: Code[AnyRef], f: Code[AnyRef] => CM[Code[AnyRef]]) => { (k: Code[AnyRef] => CM[Code[Unit]]) => f(x).flatMap(k) }
  })(aggregableHr(TTHr, aggST), unaryHr(TTHr, TUHr), aggregableHr(TUHr, aggST))

  type Id[T] = T

  def registerNumericCode[T >: Null, S >: Null](name: String, f: (Code[T], Code[T]) => Code[S])(implicit hrt: HailRep[T], hrs: HailRep[S], tti: TypeInfo[T], sti: TypeInfo[S], tct: ClassTag[T], sct: ClassTag[S]) {
    val hrboxedt = new HailRep[T] {
      def typ: Type = hrt.typ
    }
    val hrboxeds = new HailRep[S] {
      def typ: Type = hrs.typ
    }

    registerCode(name, (x: Code[T], y: Code[T]) => CM.ret(f(x, y)))

    registerCode(name, (xs: Code[IndexedSeq[T]], y: Code[T]) =>
      CM.mapIS(xs, (xOpt: Code[T]) => xOpt.mapNull((x: Code[T]) => f(x, y)))
      )(arrayHr(hrboxedt), hrt, arrayHr(hrboxeds))

    registerCode(name, (x: Code[T], ys: Code[IndexedSeq[T]]) =>
      CM.mapIS(ys, (yOpt: Code[T]) =>
        yOpt.mapNull((y: Code[T]) => f(x, y)))
      )(hrt, arrayHr(hrboxedt), arrayHr(hrboxeds))

    registerCode(name, (xs: Code[IndexedSeq[T]], ys: Code[IndexedSeq[T]]) => for (
      (stn, n) <- CM.memoize(xs.invoke[Int]("size"));
      n2 = ys.invoke[Int]("size");

      (stb, b) <- CM.memoize(Code.newArray[S](n));

      i <- CM.newLocal[Int];

      (stx, x) <- CM.memoize(xs.invoke[Int, T]("apply", i));
      (sty, y) <- CM.memoize(ys.invoke[Int, T]("apply", i));

      z = x.mapNull(y.mapNull(f(x, y)))
    ) yield Code(stn,
      (n.ceq(n2)).mux(
        Code(
          i.store(0),
          stb,
          Code.whileLoop(i < n,
            Code(stx, sty, b.update(i, z), i.store(i + 1))
          ),
          CompilationHelp.arrayToWrappedArray(b)).asInstanceOf[Code[IndexedSeq[S]]],
        Code._throw(Code.newInstance[is.hail.utils.HailException, String, Option[String], Throwable](
          s"""Cannot apply operation $name to arrays of unequal length.""".stripMargin,
          Code.invokeStatic[scala.Option[String], scala.Option[String]]("empty"),
          Code._null[Throwable])))))
  }

  def registerNumeric[T, S](name: String, f: (T, T) => S)(implicit hrt: HailRep[T], hrs: HailRep[S]) {
    val hrboxedt = new HailRep[Any] {
      def typ: Type = hrt.typ
    }
    val hrboxeds = new HailRep[Any] {
      def typ: Type = hrs.typ
    }

    register(name, f)

    register(name, (x: IndexedSeq[Any], y: T) =>
      x.map { xi =>
        if (xi == null)
          null
        else
          f(xi.asInstanceOf[T], y)
      })(arrayHr(hrboxedt), hrt, arrayHr(hrboxeds))

    register(name, (x: T, y: IndexedSeq[Any]) => y.map { yi =>
      if (yi == null)
        null
      else
        f(x, yi.asInstanceOf[T])
    })(hrt, arrayHr(hrboxedt), arrayHr(hrboxeds))

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
    })(arrayHr(hrboxedt), arrayHr(hrboxedt), arrayHr(hrboxeds))
  }

  registerMethod("toInt32", (s: String) => s.toInt)
  registerMethod("toInt64", (s: String) => s.toLong)
  registerMethod("toFloat32", (s: String) => s.toFloat)
  registerMethod("toFloat64", (s: String) => s.toDouble)
  registerMethod("toBoolean", (s: String) => s.toBoolean)

  registerMethod("toInt32", (b: Boolean) => b.toInt)
  registerMethod("toInt64", (b: Boolean) => b.toLong)
  registerMethod("toFloat32", (b: Boolean) => b.toFloat)
  registerMethod("toFloat64", (b: Boolean) => b.toDouble)

  registerMethod("sign", (x: Int) => math.signum(x))
  registerMethod("sign", (x: Long) => math.signum(x))
  registerMethod("sign", (x: Float) => math.signum(x))
  registerMethod("sign", (x: Double) => math.signum(x))
  
  def registerNumericType[T]()(implicit ev: Numeric[T], hrt: HailRep[T]) {
    // registerNumeric("+", ev.plus)
    registerNumeric("-", ev.minus)
    registerNumeric("*", ev.times)
    // registerNumeric("/", (x: T, y: T) => ev.toDouble(x) / ev.toDouble(y))

    registerMethod("abs", ev.abs _)

    register("-", ev.negate _)

    registerMethod("toInt32", ev.toInt _)
    registerMethod("toInt64", ev.toLong _)
    registerMethod("toFloat32", ev.toFloat _)
    registerMethod("toFloat64", ev.toDouble _)
  }

  registerNumeric("**", (x: Int, y: Int) => math.pow(x, y))
  registerNumeric("**", (x: Long, y: Long) => math.pow(x, y))
  registerNumeric("**", (x: Float, y: Float) => math.pow(x, y))
  registerNumeric("**", (x: Double, y: Double) => math.pow(x, y))

  registerNumericCode("/", (x: Code[java.lang.Integer], y: Code[java.lang.Integer]) => Code.boxFloat(Code.intValue(x).toF / Code.intValue(y).toF))
  registerNumericCode("/", (x: Code[java.lang.Long], y: Code[java.lang.Long]) => Code.boxFloat(Code.longValue(x).toF / Code.longValue(y).toF))
  registerNumericCode("/", (x: Code[java.lang.Float], y: Code[java.lang.Float]) => Code.boxFloat(Code.floatValue(x) / Code.floatValue(y)))
  registerNumericCode("/", (x: Code[java.lang.Double], y: Code[java.lang.Double]) => Code.boxDouble(Code.doubleValue(x).toD / Code.doubleValue(y).toD))

  registerNumericCode("+", (x: Code[java.lang.Integer], y: Code[java.lang.Integer]) => Code.boxInt(Code.intValue(x) + Code.intValue(y)))
  registerNumericCode("+", (x: Code[java.lang.Long], y: Code[java.lang.Long]) => Code.boxLong(Code.longValue(x) + Code.longValue(y)))
  registerNumericCode("+", (x: Code[java.lang.Float], y: Code[java.lang.Float]) => Code.boxFloat(Code.floatValue(x) + Code.floatValue(y)))
  registerNumericCode("+", (x: Code[java.lang.Double], y: Code[java.lang.Double]) => Code.boxDouble(Code.doubleValue(x) + Code.doubleValue(y)))

  registerNumericType[Int]()
  registerNumericType[Long]()
  registerNumericType[Float]()
  registerNumericType[Double]()

  register("==", (a: Any, b: Any) => a == b)(TTHr, TUHr, boolHr)
  register("!=", (a: Any, b: Any) => a != b)(TTHr, TUHr, boolHr)

  def registerOrderedType[T]()(implicit hrt: HailRep[T]) {
    val ord = hrt.typ.ordering

    implicit val hrboxedt = new HailRep[Any] {
      def typ: Type = hrt.typ
    }

    // register("<", ord.lt _)
    register("<=", (x: Any, y: Any) => ord.lteq(x, y))
    // register(">", ord.gt _)
    register(">=", (x: Any, y: Any) => ord.gteq(x, y))

    registerMethod("min", (x: Any, y: Any) => ord.min(x, y))
    registerMethod("max", (x: Any, y: Any) => ord.max(x, y))


    registerMethod("argmin", (a: IndexedSeq[Any]) => {
      if (a.isEmpty)
        null: java.lang.Integer
      else {
        var min = a(0)
        var minIndex = 0
        var i = 1
        while (i < a.length) {
          if (ord.lt(a(i), min)) {
            min = a(i)
            minIndex = i
          }
          i += 1
        }
        minIndex: java.lang.Integer
      }
    })

    registerMethod("argmax", (a: IndexedSeq[Any]) => {
      if (a.isEmpty)
        null: java.lang.Integer
      else {
        var max = a(0)
        var maxIndex = 0
        var i = 1
        while (i < a.length) {
          if (ord.gt(a(i), max)) {
            max = a(i)
            maxIndex = i
          }
          i += 1
        }
        maxIndex: java.lang.Integer
      }
    })

    registerMethod("uniqueMinIndex", (a: IndexedSeq[Any]) => {
      def f(i: Int, m: Any, mi: Int, count: Int): java.lang.Integer = {
        if (i == a.length) {
          assert(count >= 1)
          if (count == 1)
            mi
          else
            null
        } else if (ord.lt(a(i), m))
          f(i + 1, a(i), i, 1)
        else if (a(i) == m)
          f(i + 1, m, mi, count + 1)
        else
          f(i + 1, m, mi, count)
      }

      if (a.isEmpty)
        null
      else
        f(1, a(0), 0, 1)
    })

    registerMethod("uniqueMaxIndex", (a: IndexedSeq[Any]) => {
      def f(i: Int, m: Any, mi: Int, count: Int): java.lang.Integer = {
        if (i == a.length) {
          assert(count >= 1)
          if (count == 1)
            mi
          else
            null
        } else if (ord.gt(a(i), m))
          f(i + 1, a(i), i, 1)
        else if (a(i) == m)
          f(i + 1, m, mi, count + 1)
        else
          f(i + 1, m, mi, count)
      }

      if (a.isEmpty)
        null
      else
        f(1, a(0), 0, 1)
    })

    registerMethod("sort", (a: IndexedSeq[Any]) => a.sorted(ord.toOrdering))(arrayHr(hrboxedt), arrayHr(hrboxedt))
    registerMethod("sort", (a: IndexedSeq[Any], ascending: Boolean) =>
      a.sorted(
        (if (ascending)
           ord
         else
           ord.reverse).toOrdering)
    )(arrayHr(hrboxedt), boolHr, arrayHr(hrboxedt))

    registerLambdaMethod("sortBy", (a: IndexedSeq[Any], f: (Any) => Any) =>
      a.sortBy(f)(ord.toOrdering)
    )(arrayHr(TTHr), unaryHr(TTHr, hrboxedt), arrayHr(TTHr))

    registerLambdaMethod("sortBy", (a: IndexedSeq[Any], f: (Any) => Any, ascending: Boolean) =>
      a.sortBy(f)(
        (if (ascending)
           ord
         else
           ord.reverse).toOrdering)
    )(arrayHr(TTHr), unaryHr(TTHr, hrboxedt), boolHr, arrayHr(TTHr))
  }

  register("<", implicitly[Ordering[Boolean]].lt _)
  registerCode("<", (x: Code[java.lang.Integer], y: Code[java.lang.Integer]) => CM.ret(Code.boxBoolean(Code.intValue(x) < Code.intValue(y))))
  registerCode("<", (x: Code[java.lang.Long], y: Code[java.lang.Long]) => CM.ret(Code.boxBoolean(Code.longValue(x) < Code.longValue(y))))
  registerCode("<", (x: Code[java.lang.Float], y: Code[java.lang.Float]) => CM.ret(Code.boxBoolean(Code.floatValue(x) < Code.floatValue(y))))
  registerCode("<", (x: Code[java.lang.Double], y: Code[java.lang.Double]) => CM.ret(Code.boxBoolean(Code.doubleValue(x) < Code.doubleValue(y))))
  register("<", implicitly[Ordering[String]].lt _)

  register(">", implicitly[Ordering[Boolean]].lt _)
  registerCode(">", (x: Code[java.lang.Integer], y: Code[java.lang.Integer]) => CM.ret(Code.boxBoolean(Code.intValue(x) > Code.intValue(y))))
  registerCode(">", (x: Code[java.lang.Long], y: Code[java.lang.Long]) => CM.ret(Code.boxBoolean(Code.longValue(x) > Code.longValue(y))))
  registerCode(">", (x: Code[java.lang.Float], y: Code[java.lang.Float]) => CM.ret(Code.boxBoolean(Code.floatValue(x) > Code.floatValue(y))))
  registerCode(">", (x: Code[java.lang.Double], y: Code[java.lang.Double]) => CM.ret(Code.boxBoolean(Code.doubleValue(x) > Code.doubleValue(y))))
  register(">", implicitly[Ordering[String]].lt _)

  registerOrderedType[Boolean]()
  registerOrderedType[Int]()
  registerOrderedType[Long]()
  registerOrderedType[Float]()
  registerOrderedType[Double]()
  registerOrderedType[String]()

  register("//", (x: Int, y: Int) => java.lang.Math.floorDiv(x, y))
  register("//", (x: Long, y: Long) => java.lang.Math.floorDiv(x, y))
  register("//", (x: Float, y: Float) => math.floor(x / y).toFloat)
  register("//", (x: Double, y: Double) => math.floor(x / y))

  register("floor", (x: Float) => math.floor(x).toFloat)
  register("floor", (x: Double) => math.floor(x))

  register("ceil", (x: Float) => math.ceil(x).toFloat)
  register("ceil", (x: Double) => math.ceil(x))

  register("//", { (xs: IndexedSeq[java.lang.Integer], y: Int) =>
    val a = new Array[java.lang.Integer](xs.length)
    var i = 0
    while (i < a.length) {
      val x = xs(i)
      if (x != null)
        a(i) = java.lang.Math.floorDiv(x, y)
      i += 1
    }
    a: IndexedSeq[java.lang.Integer]
  })(arrayHr(boxedInt32Hr), int32Hr, arrayHr(boxedInt32Hr))

  register("//", { (x: Int, ys: IndexedSeq[java.lang.Integer]) =>
    val a = new Array[java.lang.Integer](ys.length)
    var i = 0
    while (i < a.length) {
      val y = ys(i)
      if (y != null)
        a(i) = java.lang.Math.floorDiv(x, y)
      i += 1
    }
    a: IndexedSeq[java.lang.Integer]
  })(int32Hr, arrayHr(boxedInt32Hr), arrayHr(boxedInt32Hr))

  register("//", { (xs: IndexedSeq[java.lang.Integer], ys: IndexedSeq[java.lang.Integer]) =>
    if (xs.length != ys.length)
      fatal(
        s"""Cannot apply operation '//' to arrays of unequal length:
           |  Left: ${ xs.length } elements, [${ xs.mkString(", ") }]
           |  Right: ${ ys.length } elements [${ ys.mkString(", ") }]""".stripMargin)

    val a = new Array[java.lang.Integer](xs.length)
    var i = 0
    while (i < a.length) {
      val x = xs(i)
      val y = ys(i)
      if (x != null && y != null)
      a(i) = java.lang.Math.floorDiv(xs(i), ys(i))
      i += 1
    }
    a: IndexedSeq[java.lang.Integer]
  })(arrayHr(boxedInt32Hr), arrayHr(boxedInt32Hr), arrayHr(boxedInt32Hr))

  register("//", { (xs: IndexedSeq[java.lang.Long], y: Long) =>
    val a = new Array[java.lang.Long](xs.length)
    var i = 0
    while (i < a.length) {
      val x = xs(i)
      if (x != null)
        a(i) = java.lang.Math.floorDiv(x, y)
      i += 1
    }
    a: IndexedSeq[java.lang.Long]
  })(arrayHr(boxedInt64Hr), int64Hr, arrayHr(boxedInt64Hr))

  register("//", { (x: Long, ys: IndexedSeq[java.lang.Long]) =>
    val a = new Array[java.lang.Long](ys.length)
    var i = 0
    while (i < a.length) {
      val y = ys(i)
      if (y != null)
        a(i) = java.lang.Math.floorDiv(x, y)
      i += 1
    }
    a: IndexedSeq[java.lang.Long]
  })(int64Hr, arrayHr(boxedInt64Hr), arrayHr(boxedInt64Hr))

  register("//", { (xs: IndexedSeq[java.lang.Long], ys: IndexedSeq[java.lang.Long]) =>
    if (xs.length != ys.length)
      fatal(
        s"""Cannot apply operation '//' to arrays of unequal length:
           |  Left: ${ xs.length } elements, [${ xs.mkString(", ") }]
           |  Right: ${ ys.length } elements [${ ys.mkString(", ") }]""".stripMargin)

    val a = new Array[java.lang.Long](xs.length)
    var i = 0
    while (i < a.length) {
      val x = xs(i)
      val y = ys(i)
      if (x != null && y != null)
        a(i) = java.lang.Math.floorDiv(xs(i), ys(i))
      i += 1
    }
    a: IndexedSeq[java.lang.Long]
  })(arrayHr(boxedInt64Hr), arrayHr(boxedInt64Hr), arrayHr(boxedInt64Hr))

  register("//", { (xs: IndexedSeq[java.lang.Float], y: Float) =>
    val a = new Array[java.lang.Float](xs.length)
    var i = 0
    while (i < a.length) {
      val x = xs(i)
      if (x != null)
        a(i) = java.lang.Math.floor(x / y).toFloat
      i += 1
    }
    a: IndexedSeq[java.lang.Float]
  })(arrayHr(boxedFloat32Hr), float32Hr, arrayHr(boxedFloat32Hr))

  register("//", { (x: Float, ys: IndexedSeq[java.lang.Float]) =>
    val a = new Array[java.lang.Float](ys.length)
    var i = 0
    while (i < a.length) {
      val y = ys(i)
      if (y != null)
        a(i) = java.lang.Math.floor(x / y).toFloat
      i += 1
    }
    a: IndexedSeq[java.lang.Float]
  })(float32Hr, arrayHr(boxedFloat32Hr), arrayHr(boxedFloat32Hr))

  register("//", { (xs: IndexedSeq[java.lang.Float], ys: IndexedSeq[java.lang.Float]) =>
    if (xs.length != ys.length)
      fatal(
        s"""Cannot apply operation '//' to arrays of unequal length:
           |  Left: ${ xs.length } elements, [${ xs.mkString(", ") }]
           |  Right: ${ ys.length } elements [${ ys.mkString(", ") }]""".stripMargin)

    val a = new Array[java.lang.Float](xs.length)
    var i = 0
    while (i < a.length) {
      val x = xs(i)
      val y = ys(i)
      if (x != null && y != null)
        a(i) = java.lang.Math.floor(x / y).toFloat
      i += 1
    }
    a: IndexedSeq[java.lang.Float]
  })(arrayHr(boxedFloat32Hr), arrayHr(boxedFloat32Hr), arrayHr(boxedFloat32Hr))

  register("//", { (xs: IndexedSeq[java.lang.Double], y: Double) =>
    val a = new Array[java.lang.Double](xs.length)
    var i = 0
    while (i < a.length) {
      val x = xs(i)
      if (x != null)
        a(i) = java.lang.Math.floor(x / y)
      i += 1
    }
    a: IndexedSeq[java.lang.Double]
  })(arrayHr(boxedFloat64Hr), float64Hr, arrayHr(boxedFloat64Hr))

  register("//", { (x: Double, ys: IndexedSeq[java.lang.Double]) =>
    val a = new Array[java.lang.Double](ys.length)
    var i = 0
    while (i < a.length) {
      val y = ys(i)
      if (y != null)
        a(i) = java.lang.Math.floor(x / y)
      i += 1
    }
    a: IndexedSeq[java.lang.Double]
  })(float64Hr, arrayHr(boxedFloat64Hr), arrayHr(boxedFloat64Hr))

  register("//", { (xs: IndexedSeq[java.lang.Double], ys: IndexedSeq[java.lang.Double]) =>
    if (xs.length != ys.length)
      fatal(
        s"""Cannot apply operation '//' to arrays of unequal length:
           |  Left: ${ xs.length } elements, [${ xs.mkString(", ") }]
           |  Right: ${ ys.length } elements [${ ys.mkString(", ") }]""".stripMargin)

    val a = new Array[java.lang.Double](xs.length)
    var i = 0
    while (i < a.length) {
      val x = xs(i)
      val y = ys(i)
      if (x != null && y != null)
        a(i) = java.lang.Math.floor(x / y)
      i += 1
    }
    a: IndexedSeq[java.lang.Double]
  })(arrayHr(boxedFloat64Hr), arrayHr(boxedFloat64Hr), arrayHr(boxedFloat64Hr))
  
  register("%", (x: Int, y: Int) => java.lang.Math.floorMod(x, y))
  register("%", (x: Long, y: Long) => java.lang.Math.floorMod(x, y))
  register("%", (x: Float, y: Float) => {
    val t = x % y
    if (x >= 0 && y > 0 || x <= 0 && y < 0 || t == 0) t else t + y
  })
  register("%", (x: Double, y: Double) => {
    val t = x % y
    if (x >= 0 && y > 0 || x <= 0 && y < 0 || t == 0) t else t + y
  })

  register("%", { (xs: IndexedSeq[java.lang.Integer], y: Int) =>
    val a = new Array[java.lang.Integer](xs.length)
    var i = 0
    while (i < a.length) {
      val x = xs(i)
      if (x != null)
        a(i) = java.lang.Math.floorMod(x, y)
      i += 1
    }
    a: IndexedSeq[java.lang.Integer]
  })(arrayHr(boxedInt32Hr), int32Hr, arrayHr(boxedInt32Hr))

  register("%", { (x: Int, ys: IndexedSeq[java.lang.Integer]) =>
    val a = new Array[java.lang.Integer](ys.length)
    var i = 0
    while (i < a.length) {
      val y = ys(i)
      if (y != null)
        a(i) = java.lang.Math.floorMod(x, y)
      i += 1
    }
    a: IndexedSeq[java.lang.Integer]
  })(int32Hr, arrayHr(boxedInt32Hr), arrayHr(boxedInt32Hr))

  register("%", { (xs: IndexedSeq[java.lang.Integer], ys: IndexedSeq[java.lang.Integer]) =>
    if (xs.length != ys.length)
      fatal(
        s"""Cannot apply operation '%' to arrays of unequal length:
           |  Left: ${ xs.length } elements, [${ xs.mkString(", ") }]
           |  Right: ${ ys.length } elements [${ ys.mkString(", ") }]""".stripMargin)

    val a = new Array[java.lang.Integer](xs.length)
    var i = 0
    while (i < a.length) {
      val x = xs(i)
      val y = ys(i)
      if (x != null && y != null)
        a(i) = java.lang.Math.floorMod(xs(i), ys(i))
      i += 1
    }
    a: IndexedSeq[java.lang.Integer]
  })(arrayHr(boxedInt32Hr), arrayHr(boxedInt32Hr), arrayHr(boxedInt32Hr))

  register("%", { (xs: IndexedSeq[java.lang.Long], y: Long) =>
    val a = new Array[java.lang.Long](xs.length)
    var i = 0
    while (i < a.length) {
      val x = xs(i)
      if (x != null)
        a(i) = java.lang.Math.floorMod(x, y)
      i += 1
    }
    a: IndexedSeq[java.lang.Long]
  })(arrayHr(boxedInt64Hr), int64Hr, arrayHr(boxedInt64Hr))

  register("%", { (x: Long, ys: IndexedSeq[java.lang.Long]) =>
    val a = new Array[java.lang.Long](ys.length)
    var i = 0
    while (i < a.length) {
      val y = ys(i)
      if (y != null)
        a(i) = java.lang.Math.floorMod(x, y)
      i += 1
    }
    a: IndexedSeq[java.lang.Long]
  })(int64Hr, arrayHr(boxedInt64Hr), arrayHr(boxedInt64Hr))

  register("%", { (xs: IndexedSeq[java.lang.Long], ys: IndexedSeq[java.lang.Long]) =>
    if (xs.length != ys.length)
      fatal(
        s"""Cannot apply operation '%' to arrays of unequal length:
           |  Left: ${ xs.length } elements, [${ xs.mkString(", ") }]
           |  Right: ${ ys.length } elements [${ ys.mkString(", ") }]""".stripMargin)

    val a = new Array[java.lang.Long](xs.length)
    var i = 0
    while (i < a.length) {
      val x = xs(i)
      val y = ys(i)
      if (x != null && y != null)
        a(i) = java.lang.Math.floorMod(xs(i), ys(i))
      i += 1
    }
    a: IndexedSeq[java.lang.Long]
  })(arrayHr(boxedInt64Hr), arrayHr(boxedInt64Hr), arrayHr(boxedInt64Hr))

  register("%", { (xs: IndexedSeq[java.lang.Float], y: Float) =>
    val a = new Array[java.lang.Float](xs.length)
    var i = 0
    while (i < a.length) {
      val x = xs(i)
      if (x != null) {
        val t = x % y
        if (x >= 0 && y > 0 || x <= 0 && y < 0 || t == 0)
          a(i) = t
        else
          a(i) = t + y
      }
      i += 1
    }
    a: IndexedSeq[java.lang.Float]
  })(arrayHr(boxedFloat32Hr), float32Hr, arrayHr(boxedFloat32Hr))

  register("%", { (x: Float, ys: IndexedSeq[java.lang.Float]) =>
    val a = new Array[java.lang.Float](ys.length)
    var i = 0
    while (i < a.length) {
      val y = ys(i)
      if (y != null) {
        val t = x % y
        if (x >= 0 && y > 0 || x <= 0 && y < 0 || t == 0)
          a(i) = t
        else
          a(i) = t + y
      }
      i += 1
    }
    a: IndexedSeq[java.lang.Float]
  })(float32Hr, arrayHr(boxedFloat32Hr), arrayHr(boxedFloat32Hr))

  register("%", { (xs: IndexedSeq[java.lang.Float], ys: IndexedSeq[java.lang.Float]) =>
    if (xs.length != ys.length)
      fatal(
        s"""Cannot apply operation '%' to arrays of unequal length:
           |  Left: ${ xs.length } elements, [${ xs.mkString(", ") }]
           |  Right: ${ ys.length } elements [${ ys.mkString(", ") }]""".stripMargin)

    val a = new Array[java.lang.Float](xs.length)
    var i = 0
    while (i < a.length) {
      val x = xs(i)
      val y = ys(i)
      if (x != null && y != null) {
        val t = x % y
        if (x >= 0 && y > 0 || x <= 0 && y < 0 || t == 0)
          a(i) = t
        else
          a(i) = t + y
      }
      i += 1
    }
    a: IndexedSeq[java.lang.Float]
  })(arrayHr(boxedFloat32Hr), arrayHr(boxedFloat32Hr), arrayHr(boxedFloat32Hr))

  register("%", { (xs: IndexedSeq[java.lang.Double], y: Double) =>
    val a = new Array[java.lang.Double](xs.length)
    var i = 0
    while (i < a.length) {
      val x = xs(i)
      if (x != null) {
        val t = x % y
        if (x >= 0 && y > 0 || x <= 0 && y < 0 || t == 0)
          a(i) = t
        else
          a(i) = t + y
      }
      i += 1
    }
    a: IndexedSeq[java.lang.Double]
  })(arrayHr(boxedFloat64Hr), float64Hr, arrayHr(boxedFloat64Hr))

  register("%", { (x: Double, ys: IndexedSeq[java.lang.Double]) =>
    val a = new Array[java.lang.Double](ys.length)
    var i = 0
    while (i < a.length) {
      val y = ys(i)
      if (y != null) {
        val t = x % y
        if (x >= 0 && y > 0 || x <= 0 && y < 0 || t == 0)
          a(i) = t
        else
          a(i) = t + y
      }
      i += 1
    }
    a: IndexedSeq[java.lang.Double]
  })(float64Hr, arrayHr(boxedFloat64Hr), arrayHr(boxedFloat64Hr))

  register("%", { (xs: IndexedSeq[java.lang.Double], ys: IndexedSeq[java.lang.Double]) =>
    if (xs.length != ys.length)
      fatal(
        s"""Cannot apply operation '%' to arrays of unequal length:
           |  Left: ${ xs.length } elements, [${ xs.mkString(", ") }]
           |  Right: ${ ys.length } elements [${ ys.mkString(", ") }]""".stripMargin)

    val a = new Array[java.lang.Double](xs.length)
    var i = 0
    while (i < a.length) {
      val x = xs(i)
      val y = ys(i)
      if (x != null && y != null) {
        val t = x % y
        if (x >= 0 && y > 0 || x <= 0 && y < 0 || t == 0)
          a(i) = t
        else
          a(i) = t + y
      }
      i += 1
    }
    a: IndexedSeq[java.lang.Double]
  })(arrayHr(boxedFloat64Hr), arrayHr(boxedFloat64Hr), arrayHr(boxedFloat64Hr))

  register("+", (x: String, y: String) => x + y)(stringHr, stringHr, stringHr)

  register("~", (s: String, t: String) => s.r.findFirstIn(t).isDefined)

  register("isnan", (d: Double) => d.isNaN)
  register("isnan", (f: Float) => f.isNaN)
  
  registerSpecial("isMissing", (g: () => Any) => g() == null)(TTHr, boolHr)
  registerSpecial("isDefined", (g: () => Any) => g() != null)(TTHr, boolHr)

  private def nullableBooleanCode(isOr: Boolean)
    (left: Code[java.lang.Boolean], right: Code[java.lang.Boolean]): Code[java.lang.Boolean] = {
    val (isZero, isNotZero) =
      if (isOr)
        (IFNE, IFEQ)
      else
        (IFEQ, IFNE)

    new Code[java.lang.Boolean] {
      // AND:
      // true true => true
      // true false => false
      // true null => null

      // null true => null
      // null false => false
      // null null => null

      // false true => false
      // false false => false
      // false null => false

      // OR:
      // true true => true
      // true false => true
      // true null => true

      // null true => true
      // null false => null
      // null null => null

      // false true => true
      // false false => false
      // false null => null
      def emit(il: Growable[AbstractInsnNode]): Unit = {
        val lnullorfalse = new LabelNode
        val ldone = new LabelNode
        val lfirst = new LabelNode
        val lsecond = new LabelNode

        left.emit(il) // L
        il += new InsnNode(DUP) // L L
        il += new JumpInsnNode(IFNULL, lnullorfalse) // L
        il += new InsnNode(DUP) // L L
        (Code._empty[java.lang.Boolean].invoke[Boolean]("booleanValue")).emit(il) // L Z
        il += new JumpInsnNode(isZero, ldone) // L

        // left = null or false
        il += lnullorfalse // L
        il += new InsnNode(DUP) // L L
        right.emit(il) // L L R
        il += new InsnNode(SWAP) // L R L
        il += new JumpInsnNode(IFNONNULL, lfirst) // L R; stack indexing is from right to left

        // left = null
        il += new InsnNode(DUP) // L R R
        il += new JumpInsnNode(IFNULL, lsecond) // L R; both are null so either one works
        il += new InsnNode(DUP) // L R R
        (Code._empty[java.lang.Boolean].invoke[Boolean]("booleanValue")).emit(il) // L R Z
        il += new JumpInsnNode(isNotZero, lsecond) // L R; stack indexing is from right to left

        il += lfirst // B A
        il += new InsnNode(SWAP) // A B
        il += lsecond // A B
        il += new InsnNode(POP) // A
        il += ldone // A
      }
    }
  }

  registerSpecialCode("||", (a: Code[java.lang.Boolean], b: Code[java.lang.Boolean]) =>
    CM.ret(nullableBooleanCode(true)(a, b)))
  registerSpecialCode("&&", (a: Code[java.lang.Boolean], b: Code[java.lang.Boolean]) =>
    CM.ret(nullableBooleanCode(false)(a, b)))

  registerMethodCode("[]", (a: Code[IndexedSeq[AnyRef]], i: Code[java.lang.Integer]) => for (
    (storei, refi) <- CM.memoize(Code.intValue(i));
    size = a.invoke[Int]("size")
  ) yield {
    Code(storei, a.invoke[Int, AnyRef]("apply", (refi >= 0).mux(refi, refi + size)))
  })(arrayHr(BoxedTTHr), boxedInt32Hr, BoxedTTHr)
  registerMethod("[]", (a: Map[Any, Any], i: Any) => a(i))(dictHr(TTHr, TUHr), TTHr, TUHr)
  registerMethod("[]", (a: String, i: Int) => (if (i >= 0) a(i) else a(a.length + i)).toString)(stringHr, int32Hr, stringHr)

  registerMethod("[:]", (a: IndexedSeq[Any]) => a)(arrayHr(TTHr), arrayHr(TTHr))
  registerMethod("[*:]", (a: IndexedSeq[Any], i: Int) => a.slice(if (i < 0) a.length + i else i, a.length))(arrayHr(TTHr), int32Hr, arrayHr(TTHr))
  registerMethod("[:*]", (a: IndexedSeq[Any], i: Int) => a.slice(0, if (i < 0) a.length + i else i))(arrayHr(TTHr), int32Hr, arrayHr(TTHr))
  registerMethod("[*:*]", (a: IndexedSeq[Any], i: Int, j: Int) => a.slice(if (i < 0) a.length + i else i, if (j < 0) a.length + j else j))(arrayHr(TTHr), int32Hr, int32Hr, arrayHr(TTHr))

  registerMethod("[:]", (a: String) => a)

  registerMethod("[*:]", (a: String, i: Int) => a.slice(if (i < 0) a.length + i else i, a.length))

  registerMethod("[:*]", (a: String, i: Int) => a.slice(0, if (i < 0) a.length + i else i))

  registerMethod("[*:*]", (a: String, i: Int, j: Int) => a.slice(if (i < 0) a.length + i else i, if (j < 0) a.length + j else j))

  registerDependentCode("str", { () =>
    val t = TT.t
    (v: Code[Any]) => CM.invokePrimitive1(t.str)(v)
  })(TTHr, stringHr)

  registerDependentCode("json", { () =>
    val t = TT.t
    (v: Code[Any]) => CM.invokePrimitive1((x: Any) => JsonMethods.compact(t.toJSON(x)))(v)
  })(TTHr, stringHr)
}
