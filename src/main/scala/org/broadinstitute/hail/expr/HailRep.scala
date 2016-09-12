package org.broadinstitute.hail.expr

import org.apache.spark.sql.catalyst.expressions.GenericRow
import org.broadinstitute.hail.utils.Interval
import org.broadinstitute.hail.variant.{AltAllele, Genotype, Locus, Variant}

trait HailRep[T] {
  def typ: Type
}

object HailRep {

  implicit object boolHr extends HailRep[Boolean] {
    def typ = TBoolean
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

  implicit object stringHr extends HailRep[String] {
    def typ = TString
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

}