package is.hail.variant

import is.hail.check.Gen
import is.hail.expr.{TInt, TStruct, Type}
import is.hail.utils._
import is.hail.variant.GenotypeType.GenotypeType
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.json4s._

//trait Call {
//  def unboxedGT: Int
//
//  def gt: Option[Int] =
//    if (unboxedGT == -1)
//      None
//    else
//      Some(unboxedGT)
//
//  def isHomRef: Boolean = unboxedGT == 0
//
//  def isHet: Boolean = unboxedGT > 0 && {
//    val p = Genotype.gtPair(unboxedGT)
//    p.j != p.k
//  }
//
//  def isHomVar: Boolean = unboxedGT > 0 && {
//    val p = Genotype.gtPair(unboxedGT)
//    p.j == p.k
//  }
//
//  def isCalledNonRef: Boolean = unboxedGT > 0
//
//  def isHetNonRef: Boolean = unboxedGT > 0 && {
//    val p = Genotype.gtPair(unboxedGT)
//    p.j > 0 && p.j != p.k
//  }
//
//  def isHetRef: Boolean = unboxedGT > 0 && {
//    val p = Genotype.gtPair(unboxedGT)
//    p.j == 0 && p.k > 0
//  }
//
//  def isNotCalled: Boolean = unboxedGT == -1
//
//  def isCalled: Boolean = unboxedGT >= 0
//
//  def gtType: GenotypeType =
//    if (isHomRef)
//      GenotypeType.HomRef
//    else if (isHet)
//      GenotypeType.Het
//    else if (isHomVar)
//      GenotypeType.HomVar
//    else {
//      assert(isNotCalled)
//      GenotypeType.NoCall
//    }
//
//  def hasNNonRefAlleles: Boolean = unboxedGT != -1
//
//  def nNonRefAlleles_ : Int = Genotype.gtPair(unboxedGT).nNonRefAlleles
//
//  def nNonRefAlleles: Option[Int] =
//    if (hasNNonRefAlleles)
//      Some(nNonRefAlleles_)
//    else
//      None
//
//  def oneHotAlleles(nAlleles: Int): Option[IndexedSeq[Int]] = {
//    gt.map { call =>
//      val gtPair = Genotype.gtPair(call)
//      val j = gtPair.j
//      val k = gtPair.k
//      new IndexedSeq[Int] {
//        def length: Int = nAlleles
//
//        def apply(idx: Int): Int = {
//          if (idx < 0 || idx >= nAlleles)
//            throw new ArrayIndexOutOfBoundsException(idx)
//          var r = 0
//          if (idx == j)
//            r += 1
//          if (idx == k)
//            r += 1
//          r
//        }
//      }
//    }
//  }
//
//  def oneHotAlleles(v: Variant): Option[IndexedSeq[Int]] = oneHotAlleles(v.nAlleles)
//
//  def oneHotGenotype(v: Variant): Option[IndexedSeq[Int]] = oneHotGenotype(v.nGenotypes)
//
//  def oneHotGenotype(nGenotypes: Int): Option[IndexedSeq[Int]] = {
//    gt.map { call =>
//      new IndexedSeq[Int] {
//        def length: Int = nGenotypes
//
//        def apply(idx: Int): Int = {
//          if (idx < 0 || idx >= nGenotypes)
//            throw new ArrayIndexOutOfBoundsException(idx)
//          if (idx == call)
//            1
//          else
//            0
//        }
//      }
//    }
//  }
//}



object Call {

  def check(unboxedGT: Int, nAlleles: Int) {
    val nGenotypes = triangle(nAlleles)
    assert(gt(unboxedGT).forall(i => i >= 0 && i < nGenotypes))
  }

  def genArb: Gen[Call] =
    for (v <- Variant.gen;
      nAlleles = v.nAlleles;
      nGenotypes = triangle(nAlleles);
      call <- Gen.choose(0, nGenotypes - 1)
    ) yield {
        check(call, nAlleles)
        call
      }

  def gt(call: java.lang.Integer): Option[Int] =
    if (call == -1)
      None
    else
      Some(call)

  def isHomRef(call: java.lang.Integer): Boolean = call == 0

  def isHet(call: java.lang.Integer): Boolean = call > 0 && {
    val p = Genotype.gtPair(call)
    p.j != p.k
  }

  def isHomVar(call: java.lang.Integer): Boolean = call > 0 && {
    val p = Genotype.gtPair(call)
    p.j == p.k
  }

  def isCalledNonRef(call: java.lang.Integer): Boolean = call > 0

  def isHetNonRef(call: java.lang.Integer): Boolean = call > 0 && {
    val p = Genotype.gtPair(call)
    p.j > 0 && p.j != p.k
  }

  def isHetRef(call: java.lang.Integer): Boolean = call > 0 && {
    val p = Genotype.gtPair(call)
    p.j == 0 && p.k > 0
  }

  def isNotCalled(call: java.lang.Integer): Boolean = call == -1

  def isCalled(call: java.lang.Integer): Boolean = call >= 0

  def gtType(call: java.lang.Integer): GenotypeType =
    if (isHomRef(call))
      GenotypeType.HomRef
    else if (isHet(call))
      GenotypeType.Het
    else if (isHomVar(call))
      GenotypeType.HomVar
    else {
      assert(isNotCalled(call))
      GenotypeType.NoCall
    }

  def hasNNonRefAlleles(call: java.lang.Integer): Boolean = call != -1

  def nNonRefAlleles_(call: java.lang.Integer) : Int = Genotype.gtPair(call).nNonRefAlleles

  def nNonRefAlleles(call: java.lang.Integer): Option[Int] =
    if (hasNNonRefAlleles(call))
      Some(nNonRefAlleles_(call))
    else
      None

  def oneHotAlleles(call: java.lang.Integer, nAlleles: Int): Option[IndexedSeq[Int]] = {
    gt(call).map { call => // FIXME???
      val gtPair = Genotype.gtPair(call)
      val j = gtPair.j
      val k = gtPair.k
      new IndexedSeq[Int] {
        def length: Int = nAlleles

        def apply(idx: Int): Int = {
          if (idx < 0 || idx >= nAlleles)
            throw new ArrayIndexOutOfBoundsException(idx)
          var r = 0
          if (idx == j)
            r += 1
          if (idx == k)
            r += 1
          r
        }
      }
    }
  }

  def oneHotGenotype(call: java.lang.Integer, nGenotypes: Int): Option[IndexedSeq[Int]] = {
    gt(call).map { call =>
      new IndexedSeq[Int] {
        def length: Int = nGenotypes

        def apply(idx: Int): Int = {
          if (idx < 0 || idx >= nGenotypes)
            throw new ArrayIndexOutOfBoundsException(idx)
          if (idx == call)
            1
          else
            0
        }
      }
    }
  }
}
