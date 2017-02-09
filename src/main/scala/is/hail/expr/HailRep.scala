package is.hail.expr

import is.hail.utils.Interval
import is.hail.variant.{AltAllele, Genotype, Locus, Variant}

trait HailRep[T] { self =>
  def typ: Type
}

trait HailRepFunctions {

  implicit object boolHr extends HailRep[Boolean] {
    def typ = TBoolean
  }

  object charHr extends HailRep[String] {
    def typ = TChar
  }

  implicit object intHr extends HailRep[Int] {
    def typ = TInt
  }

  implicit object longHr extends HailRep[Long] {
    def typ = TLong
  }

  implicit object floatHr extends HailRep[Float] {
    def typ = TFloat
  }

  implicit object doubleHr extends HailRep[Double] {
    def typ = TDouble
  }

  object boxedboolHr extends HailRep[Any] {
    def typ = TBoolean
  }

  object boxedintHr extends HailRep[Any] {
    def typ = TInt
  }

  object boxedlongHr extends HailRep[Any] {
    def typ = TLong
  }

  object boxedfloatHr extends HailRep[Any] {
    def typ = TFloat
  }

  object boxeddoubleHr extends HailRep[Any] {
    def typ = TDouble
  }

  implicit object stringHr extends HailRep[String] {
    def typ = TString
  }

  // not implicit to make stringHr the default
  object sampleHr extends HailRep[String] {
    def typ = TSample
  }

  implicit object genotypeHr extends HailRep[Genotype] {
    def typ = TGenotype
  }

  implicit object variantHr extends HailRep[Variant] {
    def typ = TVariant
  }

  implicit object locusHr extends HailRep[Locus] {
    def typ = TLocus
  }

  implicit object altAlleleHr extends HailRep[AltAllele] {
    def typ = TAltAllele
  }

  implicit object locusIntervalHr extends HailRep[Interval[Locus]] {
    def typ = TInterval
  }

  implicit def arrayHr[T](implicit hrt: HailRep[T]) = new HailRep[IndexedSeq[T]] {
    def typ = TArray(hrt.typ)
  }

  implicit def setHr[T](implicit hrt: HailRep[T]) = new HailRep[Set[T]] {
    def typ = TSet(hrt.typ)
  }

  implicit def dictHr[K, V](implicit hrt: HailRep[K], hrt2: HailRep[V]) = new HailRep[Map[K, V]] {
    def typ = TDict(hrt.typ, hrt2.typ)
  }

  implicit def unaryHr[T, U](implicit hrt: HailRep[T], hru: HailRep[U]) = new HailRep[(Any) => Any] {
    def typ = TFunction(Seq(hrt.typ), hru.typ)
  }

  def aggregableHr[T](implicit hrt: HailRep[T]) = new HailRep[T] {
    def typ = TAggregable(hrt.typ)
  }

  def aggregableHr[T](hrt: HailRep[T], b: Box[SymbolTable]) = new HailRep[T] {
    def typ = TAggregableVariable(hrt.typ, b)
  }
}