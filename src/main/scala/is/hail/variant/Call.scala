package is.hail.variant

import is.hail.check.Gen
import is.hail.utils._
import is.hail.variant.GenotypeType.GenotypeType

object Call {

  def apply(call: Int) = {
    require(call > -1, s"Invalid Call input `$call'. Must be >= -1.")
    call
  }

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

  def nNonRefAlleles_(call: java.lang.Integer): Int = Genotype.gtPair(call).nNonRefAlleles

  def nNonRefAlleles(call: java.lang.Integer): Option[Int] =
    if (hasNNonRefAlleles(call))
      Some(nNonRefAlleles_(call))
    else
      None

  def oneHotAlleles(call: java.lang.Integer, nAlleles: Int): Option[IndexedSeq[Int]] = {
    gt(call).map { call =>
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
