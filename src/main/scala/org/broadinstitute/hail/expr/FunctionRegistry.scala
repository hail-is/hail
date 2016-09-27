package org.broadinstitute.hail.expr

import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.stats._
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.{AltAllele, Genotype, Locus, Variant}
import org.broadinstitute.hail.expr.HailRep._

import scala.collection.mutable

object FunctionRegistry {

  private val registry = new mutable.HashMap[(String, TypeTag), Fun]

  private def lookup(name: String, typ: TypeTag): Option[Fun] =
    registry.get((name, typ))

  private def bind(name: String, typ: TypeTag, f: Fun) = {
    lookup(name, typ) match {
      case Some(existingBinding) =>
        throw new RuntimeException(s"The name, ${ name }, with type, ${ typ }, is already bound as ${ existingBinding }")
      case None =>
        registry.put((name, typ), f)
    }
  }

  def lookupField(ec: EvalContext)(typ: BaseType, name: String)(xAst: AST): Option[() => Any] =
    lookup(name, MethodType(typ)).map {
      case f: UnaryFun[_, _] => AST.evalCompose(ec, xAst)(f)
      case f: OptionUnaryFun[_, _] => AST.evalFlatCompose(ec, xAst)(f)
      case f =>
        throw new RuntimeException(s"Internal hail error, bad binding in function registry for `$name' with argument type $typ: $f")
    }

  def lookupFieldType(typ: BaseType, name: String): Option[Type] =
    lookup(name, MethodType(typ)).map(f => f.retType)

  def lookupFun(ec: EvalContext)(name: String, typs: Seq[BaseType])(args: Seq[AST]): Option[() => Any] = {
    require(typs.length == args.length)

    lookup(name, FunType(typs: _*)).map { fn => (fn, typs) match {
      case (f: UnaryFun[_, _], Seq(x)) =>
        AST.evalCompose(ec, args(0))(f)
      case (f: OptionUnaryFun[_, _], Seq(x)) =>
        AST.evalFlatCompose(ec, args(0))(f)
      case (f: BinaryFun[_, _, _], Seq(x, y)) =>
        AST.evalCompose(ec, args(0), args(1))(f)
      case (f: Arity3Fun[_, _, _, _], Seq(x, y, z)) =>
        AST.evalCompose(ec, args(0), args(1), args(2))(f)
      case (f: Arity4Fun[_, _, _, _, _], Seq(x, y, z, a)) =>
        AST.evalCompose(ec, args(0), args(1), args(2), args(3))(f)
      case _ =>
        throw new RuntimeException(s"Internal hail error, bad binding in function registry for `$name' with argument types $typs: $fn")
    }
    }
  }

  def lookupFunReturnType(name: String, typs: Seq[BaseType]): Option[Type] =
    lookup(name, FunType(typs: _*)).map(_.retType)

  def registerField[T, U](name: String, impl: T => U)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, MethodType(hrt.typ), UnaryFun[T, U](hru.typ, impl))
  }

  def register[T, U](name: String, impl: T => U)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, FunType(hrt.typ), UnaryFun[T, U](hru.typ, impl))
  }

  def registerOptionField[T, U](name: String, impl: T => Option[U])
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, MethodType(hrt.typ), OptionUnaryFun[T, U](hru.typ, impl))
  }

  def registerOption[T, U](name: String, impl: T => Option[U])
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, FunType(hrt.typ), OptionUnaryFun[T, U](hru.typ, impl))
  }

  def registerUnaryNAFilteredCollectionField[T, U](name: String, impl: TraversableOnce[T] => U)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    bind(name, MethodType(TArray(hrt.typ)), UnaryFun[IndexedSeq[_], U](hru.typ, { (ts: IndexedSeq[_]) =>
      impl(ts.filter(t => t != null).map(_.asInstanceOf[T]))
    }))
    bind(name, MethodType(TSet(hrt.typ)), UnaryFun[Set[_], U](hru.typ, { (ts: Set[_]) =>
      impl(ts.filter(t => t != null).map(_.asInstanceOf[T]))
    }))
  }

  def register[T, U, V](name: String, impl: (T, U) => V)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    bind(name, FunType(hrt.typ, hru.typ), BinaryFun[T, U, V](hrv.typ, impl))
  }

  def register[T, U, V, W](name: String, impl: (T, U, V) => W)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    bind(name, FunType(hrt.typ, hru.typ, hrv.typ), Arity3Fun[T, U, V, W](hrw.typ, impl))
  }

  def register[T, U, V, W, X](name: String, impl: (T, U, V, W) => X)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W], hrx: HailRep[X]) = {
    bind(name, FunType(hrt.typ, hru.typ, hrv.typ, hrw.typ), Arity4Fun[T, U, V, W, X](hrx.typ, impl))
  }

  def registerAnn[T](name: String, t: TStruct, impl: T => Annotation)
    (implicit hrt: HailRep[T]) = {
    register(name, impl)(hrt, new HailRep[Annotation] {def typ = t })
  }

  def registerAnn[T, U](name: String, t: TStruct, impl: (T, U) => Annotation)
    (implicit hrt: HailRep[T], hru: HailRep[U]) = {
    register(name, impl)(hrt, hru, new HailRep[Annotation] {def typ = t })
  }

  def registerAnn[T, U, V](name: String, t: TStruct, impl: (T, U, V) => Annotation)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V]) = {
    register(name, impl)(hrt, hru, hrv, new HailRep[Annotation] {def typ = t })
  }

  def registerAnn[T, U, V, W](name: String, t: TStruct, impl: (T, U, V, W) => Annotation)
    (implicit hrt: HailRep[T], hru: HailRep[U], hrv: HailRep[V], hrw: HailRep[W]) = {
    register(name, impl)(hrt, hru, hrv, hrw, new HailRep[Annotation] {def typ = t })
  }

  registerOptionField("gt", { (x: Genotype) => x.gt })
  registerOptionField("gtj", { (x: Genotype) => x.gt.map(gtx => Genotype.gtPair(gtx).j) })
  registerOptionField("gtk", { (x: Genotype) => x.gt.map(gtx => Genotype.gtPair(gtx).k) })
  registerOptionField("ad", { (x: Genotype) => x.ad.map(a => a: IndexedSeq[Int]) })
  registerOptionField("dp", { (x: Genotype) => x.dp })
  registerOptionField("od", { (x: Genotype) => x.od })
  registerOptionField("gq", { (x: Genotype) => x.gq })
  registerOptionField("pl", { (x: Genotype) => x.pl.map(a => a: IndexedSeq[Int]) })
  registerOptionField("dosage", { (x: Genotype) => x.dosage.map(a => a: IndexedSeq[Double]) })
  registerField("isHomRef", { (x: Genotype) => x.isHomRef })
  registerField("isHet", { (x: Genotype) => x.isHet })
  registerField("isHomVar", { (x: Genotype) => x.isHomVar })
  registerField("isCalledNonRef", { (x: Genotype) => x.isCalledNonRef })
  registerField("isHetNonRef", { (x: Genotype) => x.isHetNonRef })
  registerField("isHetRef", { (x: Genotype) => x.isHetRef })
  registerField("isCalled", { (x: Genotype) => x.isCalled })
  registerField("isNotCalled", { (x: Genotype) => x.isNotCalled })
  registerOptionField("nNonRefAlleles", { (x: Genotype) => x.nNonRefAlleles })
  registerOptionField("pAB", { (x: Genotype) => x.pAB() })
  registerOptionField("fractionReadsRef", { (x: Genotype) => x.fractionReadsRef() })
  registerField("fakeRef", { (x: Genotype) => x.fakeRef })
  registerField("isDosage", { (x: Genotype) => x.isDosage })
  registerField("contig", { (x: Variant) => x.contig })
  registerField("start", { (x: Variant) => x.start })
  registerField("ref", { (x: Variant) => x.ref })
  registerField("altAlleles", { (x: Variant) => x.altAlleles })
  registerField("nAltAlleles", { (x: Variant) => x.nAltAlleles })
  registerField("nAlleles", { (x: Variant) => x.nAlleles })
  registerField("isBiallelic", { (x: Variant) => x.isBiallelic })
  registerField("nGenotypes", { (x: Variant) => x.nGenotypes })
  registerField("inXPar", { (x: Variant) => x.inXPar })
  registerField("inYPar", { (x: Variant) => x.inYPar })
  registerField("inXNonPar", { (x: Variant) => x.inXNonPar })
  registerField("inYNonPar", { (x: Variant) => x.inYNonPar })
  // assumes biallelic
  registerField("alt", { (x: Variant) => x.alt })
  registerField("altAllele", { (x: Variant) => x.altAllele })
  registerField("locus", { (x: Variant) => x.locus })
  registerField("contig", { (x: Locus) => x.contig })
  registerField("position", { (x: Locus) => x.position })
  registerField("start", { (x: Interval[Locus]) => x.start })
  registerField("end", { (x: Interval[Locus]) => x.end })
  registerField("ref", { (x: AltAllele) => x.ref })
  registerField("alt", { (x: AltAllele) => x.alt })
  registerField("isSNP", { (x: AltAllele) => x.isSNP })
  registerField("isMNP", { (x: AltAllele) => x.isMNP })
  registerField("isIndel", { (x: AltAllele) => x.isIndel })
  registerField("isInsertion", { (x: AltAllele) => x.isInsertion })
  registerField("isDeletion", { (x: AltAllele) => x.isDeletion })
  registerField("isComplex", { (x: AltAllele) => x.isComplex })
  registerField("isTransition", { (x: AltAllele) => x.isTransition })
  registerField("isTransversion", { (x: AltAllele) => x.isTransversion })

  registerField("toInt", { (x: Int) => x })
  registerField("toLong", { (x: Int) => x.toLong })
  registerField("toFloat", { (x: Int) => x.toFloat })
  registerField("toDouble", { (x: Int) => x.toDouble })

  registerField("toInt", { (x: Long) => x.toInt })
  registerField("toLong", { (x: Long) => x })
  registerField("toFloat", { (x: Long) => x.toFloat })
  registerField("toDouble", { (x: Long) => x.toDouble })

  registerField("toInt", { (x: Float) => x.toInt })
  registerField("toLong", { (x: Float) => x.toLong })
  registerField("toFloat", { (x: Float) => x })
  registerField("toDouble", { (x: Float) => x.toDouble })

  registerField("toInt", { (x: Double) => x.toInt })
  registerField("toLong", { (x: Double) => x.toLong })
  registerField("toFloat", { (x: Double) => x.toFloat })
  registerField("toDouble", { (x: Double) => x })

  registerField("toInt", { (x: String) => x.toInt })
  registerField("toLong", { (x: String) => x.toLong })
  registerField("toFloat", { (x: String) => x.toFloat })
  registerField("toDouble", { (x: String) => x.toDouble })

  registerField("abs", { (x: Int) => x.abs })
  registerField("abs", { (x: Long) => x.abs })
  registerField("abs", { (x: Float) => x.abs })
  registerField("abs", { (x: Double) => x.abs })

  registerField("signum", { (x: Int) => x.signum })
  registerField("signum", { (x: Long) => x.signum })
  registerField("signum", { (x: Float) => x.signum })
  registerField("signum", { (x: Double) => x.signum })

  registerField("length", { (x: String) => x.length })

  registerUnaryNAFilteredCollectionField("sum", { (x: TraversableOnce[Int]) => x.sum })
  registerUnaryNAFilteredCollectionField("sum", { (x: TraversableOnce[Long]) => x.sum })
  registerUnaryNAFilteredCollectionField("sum", { (x: TraversableOnce[Float]) => x.sum })
  registerUnaryNAFilteredCollectionField("sum", { (x: TraversableOnce[Double]) => x.sum })

  registerUnaryNAFilteredCollectionField("min", { (x: TraversableOnce[Int]) => x.min })
  registerUnaryNAFilteredCollectionField("min", { (x: TraversableOnce[Long]) => x.min })
  registerUnaryNAFilteredCollectionField("min", { (x: TraversableOnce[Float]) => x.min })
  registerUnaryNAFilteredCollectionField("min", { (x: TraversableOnce[Double]) => x.min })

  registerUnaryNAFilteredCollectionField("max", { (x: TraversableOnce[Int]) => x.max })
  registerUnaryNAFilteredCollectionField("max", { (x: TraversableOnce[Long]) => x.max })
  registerUnaryNAFilteredCollectionField("max", { (x: TraversableOnce[Float]) => x.max })
  registerUnaryNAFilteredCollectionField("max", { (x: TraversableOnce[Double]) => x.max })

  register("range", { (x: Int) => (0 until x).toArray: IndexedSeq[Int] })
  register("range", { (x: Int, y: Int) => (x until y).toArray: IndexedSeq[Int] })
  register("Variant", { (x: String) =>
    val Array(chr, pos, ref, alts) = x.split(":")
    Variant(chr, pos.toInt, ref, alts.split(","))
  })
  register("Variant", { (x: String, y: Int, z: String, a: String) => Variant(x, y, z, a) })
  register("Variant", { (x: String, y: Int, z: String, a: IndexedSeq[String]) => Variant(x, y, z, a.toArray) })

  register("Locus", { (x: String) =>
    val Array(chr, pos) = x.split(":")
    Locus(chr, pos.toInt)
  })
  register("Locus", { (x: String, y: Int) => Locus(x, y) })
  register("Interval", { (x: Locus, y: Locus) => Interval(x, y) })
  registerAnn("hwe", TStruct(("rExpectedHetFrequency", TDouble), ("pHWE", TDouble)), { (nHomRef: Int, nHet: Int, nHomVar: Int) =>
    if (nHomRef < 0 || nHet < 0 || nHomVar < 0)
      fatal(s"got invalid (negative) argument to function `hwe': hwe($nHomRef, $nHet, $nHomVar)")
    val n = nHomRef + nHet + nHomVar
    val nAB = nHet
    val nA = nAB + 2 * nHomRef.min(nHomVar)

    val LH = LeveneHaldane(n, nA)
    Annotation(divOption(LH.getNumericalMean, n).orNull, LH.exactMidP(nAB))
  })
  registerAnn("fet", TStruct(("pValue", TDouble), ("oddsRatio", TDouble), ("ci95Lower", TDouble), ("ci95Upper", TDouble)), { (c1: Int, c2: Int, c3: Int, c4: Int) =>
    if (c1 < 0 || c2 < 0 || c3 < 0 || c4 < 0)
      fatal(s"got invalid argument to function `fet': fet($c1, $c2, $c3, $c4)")
    val fet = FisherExactTest(c1, c2, c3, c4)
    Annotation(fet(0).orNull, fet(1).orNull, fet(2).orNull, fet(3).orNull)
  })
  // NB: merge takes two structs, how do I deal with structs?
  register("exp", { (x: Int) => math.exp(x) })
  register("exp", { (x: Long) => math.exp(x) })
  register("exp", { (x: Float) => math.exp(x) })
  register("exp", { (x: Double) => math.exp(x) })
  register("log10", { (x: Int) => math.log10(x) })
  register("log10", { (x: Long) => math.log10(x) })
  register("log10", { (x: Float) => math.log10(x) })
  register("log10", { (x: Double) => math.log10(x) })
  register("sqrt", { (x: Int) => math.sqrt(x) })
  register("sqrt", { (x: Long) => math.sqrt(x) })
  register("sqrt", { (x: Float) => math.sqrt(x) })
  register("sqrt", { (x: Double) => math.sqrt(x) })
  register("log", { (x: Int) => math.log(x) })
  register("log", { (x: Long) => math.log(x) })
  register("log", { (x: Float) => math.log(x) })
  register("log", { (x: Double) => math.log(x) })

  register("pcoin", { (p: Double) => math.random < p })
}
