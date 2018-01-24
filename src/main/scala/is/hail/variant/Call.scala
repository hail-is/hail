package is.hail.variant

import java.io.Serializable

import is.hail.check.Gen
import is.hail.utils._

object Call extends Serializable {

  def apply(call: Int): Call = {
    require(call >= 0, s"Call must be null or >= 0. Found ${ call }.")
    call
  }

  def toString(call: Call): String = {
      val p = Genotype.gtPair(call)
      s"${ p.j }/${ p.k }"
    }

  def check(call: Call, nAlleles: Int) {
    val nGenotypes = triangle(nAlleles)
    assert(call >= 0 && call < nGenotypes, s"Invalid genotype found `$call' for number of alleles equal to `$nAlleles'.")
  }

  def genArb: Gen[Call] =
    for {
      v <- Variant.gen
      nAlleles = v.nAlleles
      nGenotypes = triangle(nAlleles)
      call <- Gen.choose(0, nGenotypes - 1)
    } yield {
      check(call, nAlleles)
      call
    }

  def genNonmissingValue: Gen[Call] =
    for {
      v <- Variant.gen
      nAlleles = v.nAlleles
      nGenotypes = triangle(nAlleles)
      call <- Gen.choose(0, nGenotypes - 1)
    } yield {
      check(call, nAlleles)
      call
    }

  def isHomRef(call: Call): Boolean = call == 0

  def isHet(call: Call): Boolean = call > 0 && {
    val p = Genotype.gtPair(call)
    p.j != p.k
  }

  def isHomVar(call: Call): Boolean = call > 0 && {
    val p = Genotype.gtPair(call)
    p.j == p.k
  }

  def isNonRef(call: Call): Boolean = call > 0

  def isHetNonRef(call: Call): Boolean = call > 0 && {
    val p = Genotype.gtPair(call)
    p.j > 0 && p.j != p.k
  }

  def isHetRef(call: Call): Boolean = call > 0 && {
    val p = Genotype.gtPair(call)
    p.j == 0 && p.k > 0
  }

  def gtj(call: Call): Int = Genotype.gtPair(call).j

  def gtk(call: Call): Int = Genotype.gtPair(call).k

  def nNonRefAlleles(call: Call): Int = Genotype.gtPair(call).nNonRefAlleles

  def oneHotAlleles(call: Call, nAlleles: Int): IndexedSeq[Int] = {
    val gtPair = Genotype.gtPair(call)
    val j = gtPair.j
    val k = gtPair.k
    new IndexedSeq[Int] with Serializable {
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

  def oneHotGenotype(call: Call, nGenotypes: Int): IndexedSeq[Int] = {
    new IndexedSeq[Int] with Serializable {
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

  def oneHotAlleles(call: Call, v: Variant): IndexedSeq[Int] = oneHotAlleles(call, v.nAlleles)

  def oneHotGenotype(call: Call, v: Variant): IndexedSeq[Int] = oneHotGenotype(call, v.nGenotypes)
}
