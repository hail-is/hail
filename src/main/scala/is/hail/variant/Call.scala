package is.hail.variant

import is.hail.check.Gen
import is.hail.utils._
import is.hail.variant.GenotypeType.GenotypeType

object Call {

  def apply(call: java.lang.Integer): Call = {
    require(call == null || call >= 0)
    call
  }

  def apply(g: Genotype): Call = {
    val call = g.unboxedGT
    if (call == -1) null else box(call)
  }

  def check(call: Call, nAlleles: Int) {
    val nGenotypes = triangle(nAlleles)
    assert(gt(call).forall(i => i >= 0 && i < nGenotypes))
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

  def gt(call: Call): Option[Int] = Option(call)

  def isHomRef(call: Call): Boolean = call == 0

  def isHet(call: Call): Boolean = call > 0 && {
    val p = Genotype.gtPair(call)
    p.j != p.k
  }

  def isHomVar(call: Call): Boolean = isCalled(call) && call > 0 && {
    val p = Genotype.gtPair(call)
    p.j == p.k
  }

  def isCalledNonRef(call: Call): Boolean = isCalled(call) && call > 0

  def isHetNonRef(call: Call): Boolean = isCalled(call) && call > 0 && {
    val p = Genotype.gtPair(call)
    p.j > 0 && p.j != p.k
  }

  def isHetRef(call: Call): Boolean = isCalled(call) && call > 0 && {
    val p = Genotype.gtPair(call)
    p.j == 0 && p.k > 0
  }

  def isNotCalled(call: Call): Boolean = call == null

  def isCalled(call: Call): Boolean = !isNotCalled(call)

  def gtType(call: Call): GenotypeType =
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

  def hasNNonRefAlleles(call: Call): Boolean = isCalled(call)

  def nNonRefAlleles_(call: Call): Int = Genotype.gtPair(call).nNonRefAlleles

  def nNonRefAlleles(call: Call): Option[Int] =
    if (hasNNonRefAlleles(call))
      Some(nNonRefAlleles_(call))
    else
      None

  def oneHotAlleles(call: Call, nAlleles: Int): Option[IndexedSeq[Int]] = {
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

  def oneHotGenotype(call: Call, nGenotypes: Int): Option[IndexedSeq[Int]] = {
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
