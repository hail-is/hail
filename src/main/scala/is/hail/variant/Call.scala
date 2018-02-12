package is.hail.variant

import java.io.Serializable

import is.hail.check.Gen
import is.hail.expr.Parser
import is.hail.utils._

import scala.annotation.switch
import scala.collection.JavaConverters._
import scala.language.implicitConversions

object Call0 {
  def apply(phased: Boolean = false): Call = {
    Call(0, phased, ploidy = 0)
  }
}

object Call1 {
  def apply(aj: Int, phased: Boolean = false): Call = {
    if (aj < 0)
      fatal(s"allele index must be >= 0. Found $aj.")
    Call(aj, phased, ploidy = 1)
  }
}

object Call2 {
  def fromUnphasedDiploidGtIndex(gt: Int): Call = {
    if (gt < 0)
      fatal(s"gt must be >= 0. Found $gt.")
    Call(gt, phased = false, ploidy = 2)
  }

  def apply(aj: Int, ak: Int, phased: Boolean = false): Call = {
    if (aj < 0 || ak < 0)
      fatal(s"allele indices must be >= 0. Found j=$aj and k=$ak.")

    val alleleRepr =
      if (phased)
        Genotype.diploidGtIndex(aj, aj + ak)
      else
        Genotype.diploidGtIndexWithSwap(aj, ak)

    Call(alleleRepr, phased, ploidy = 2)
  }
}

object CallN {
  def apply(alleles: java.util.ArrayList[Int], phased: Boolean): Call = apply(alleles.asScala.toArray, phased)

  def apply(alleles: Array[Int], phased: Boolean = false): Call = {
    val ploidy = alleles.length
    (ploidy: @switch) match {
      case 0 => Call0(phased)
      case 1 => Call1(alleles(0), phased)
      case 2 => Call2(alleles(0), alleles(1), phased)
      case _ => throw new UnsupportedOperationException
    }
  }
}

object Call extends Serializable {
  def apply(ar: Int, phased: Boolean, ploidy: Int): Call = {
    if (ploidy < 0 || ploidy > 2)
      fatal(s"invalid ploidy: $ploidy. Only support ploidy in range [0, 2]")
    if (ar < 0)
      fatal(s"invalid allele representation: $ar. Must be positive.")

    var c = 0
    c |= phased.toInt

    if (ploidy > 2)
      c |= (3 << 1)
    else
      c |= (ploidy << 1)

    if ((ar >>> 29) != 0)
      fatal(s"invalid allele representation: $ar. Max value is 2^29 - 1")

    c |= ar << 3
    c
  }

  def isPhased(c: Call): Boolean = (c & 0x1) == 1

  def isHaploid(c: Call): Boolean = ploidy(c) == 1

  def isDiploid(c: Call): Boolean = ploidy(c) == 2

  def isUnphasedDiploid(c: Call): Boolean = (c & 0x7) == 4

  def isPhasedDiploid(c: Call): Boolean = (c & 0x7) == 5

  def ploidy(c: Call): Int = (c >>> 1) & 0x3

  def alleleRepr(c: Call): Int = c >>> 3

  def allelePair(c: Call): AllelePair = {
    if (!isDiploid(c))
      fatal(s"invalid ploidy: ${ ploidy(c) }. Only support ploidy == 2")

    if (isPhased(c)) {
        val p = Genotype.allelePair(alleleRepr(c))
        AllelePair(p.j, p.k - p.j)
    } else
      Genotype.allelePair(alleleRepr(c))
  }

  def unphasedDiploidGtIndex(c: Call): Int = {
    if (!isUnphasedDiploid(c))
      fatal(s"Only support ploidy == 2 and unphased. Found ${ Call.toString(c) }.")
    alleleRepr(c)
  }

  def alleles(c: Call): Array[Int] = {
    (ploidy(c): @switch) match {
      case 0 => Array.empty[Int]
      case 1 => Array(alleleByIndex(c, 0))
      case 2 => allelePair(c).alleleIndices
      case _ => throw new UnsupportedOperationException
    }
  }

  def alleleByIndex(c: Call, i: Int): Int = {
    (ploidy(c): @switch) match {
      case 0 => throw new UnsupportedOperationException
      case 1 =>
        require(i == 0)
        alleleRepr(c)
      case 2 =>
        require (i == 0 || i == 1)
        val p = allelePair(c)
        if (i == 0) p.j else p.k
      case _ =>
        require(i >= 0 && i < ploidy(c))
        alleles(c)(i)
    }
  }

  def parse(s: String): Call = Parser.parseCall(s)

  def toString(c: Call): String = {
    val phased = isPhased(c)
    val sep = if (phased) "|" else "/"

    (ploidy(c): @switch) match {
      case 0 => if (phased) "|-" else "-"
      case 1 =>
        val a = alleleByIndex(c, 0)
        if (phased) s"|$a" else s"$a"
      case 2 =>
        val p = allelePair(c)
        s"${ p.j }$sep${ p.k }"
      case _ =>
        alleles(c).mkString(sep)
    }
  }

  def vcfString(c: Call, sb: StringBuilder): Unit = {
    val phased = isPhased(c)
    val sep = if (phased) "|" else "/"

    (ploidy(c): @switch) match {
      case 0 =>
        throw new UnsupportedOperationException("VCF spec does not support 0-ploid calls.")
      case 1 =>
        if (phased)
          throw new UnsupportedOperationException("VCF spec does not support phased haploid calls.")
        else
          sb.append(alleleByIndex(c, 0))
      case 2 =>
        val p = allelePair(c)
        sb.append(p.j)
        sb.append(sep)
        sb.append(p.k)
      case _ =>
        var i = 0
        val nAlleles = ploidy(c)
        while (i < nAlleles) {
          sb.append(alleleByIndex(c, i))
          if (i != nAlleles - 1)
            sb.append(sep)
          i += 1
        }
    }
  }

  def isHomRef(c: Call): Boolean = {
    (ploidy(c): @switch) match {
      case 0 => false
      case 1 | 2 => alleleRepr(c) == 0
      case _ => alleles(c).forall(_ == 0)
    }
  }

  def isHet(c: Call): Boolean = {
    (ploidy(c): @switch) match {
      case 0 | 1 => false
      case 2 => alleleRepr(c) > 0 && {
        val p = allelePair(c)
        p.j != p.k
      }
      case _ => throw new UnsupportedOperationException
    }
  }

  def isHomVar(c: Call): Boolean = {
    (ploidy(c): @switch) match {
      case 0 => false
      case 1 => alleleRepr(c) > 0
      case 2 => alleleRepr(c) > 0 && {
        val p = allelePair(c)
        p.j == p.k
      }
      case _ => throw new UnsupportedOperationException
    }
  }

  def isNonRef(c: Call): Boolean = {
    (ploidy(c): @switch) match {
      case 0 => false
      case 1 | 2 => alleleRepr(c) > 0
      case _ => alleles(c).exists(_ != 0)
    }
  }

  def isHetNonRef(c: Call): Boolean = {
    (ploidy(c): @switch) match {
      case 0 | 1 => false
      case 2 => alleleRepr(c) > 0 && {
        val p = allelePair(c)
        p.j > 0 && p.k > 0 && p.k != p.j
      }
      case _ => throw new UnsupportedOperationException
    }
  }

  def isHetRef(c: Call): Boolean = {
    (ploidy(c): @switch) match {
      case 0 | 1 => false
      case 2 => alleleRepr(c) > 0 && {
        val p = allelePair(c)
        (p.j == 0 && p.k > 0) || (p.k == 0 && p.j > 0)
      }
      case _ => throw new UnsupportedOperationException
    }
  }

  def nNonRefAlleles(c: Call): Int = {
    (ploidy(c): @switch) match {
      case 0 => 0
      case 1 => (alleleRepr(c) > 0).toInt
      case 2 => allelePair(c).nNonRefAlleles
      case _ => alleles(c).count(_ != 0)
    }
  }

  def oneHotAlleles(c: Call, nAlleles: Int): IndexedSeq[Int] = {
    var j = 0
    var k = 0

    if (ploidy(c) == 2) {
      val p = allelePair(c)
      j = p.j
      k = p.k
    }

    new IndexedSeq[Int] with Serializable {
      def length: Int = nAlleles

      def apply(idx: Int): Int = {
        if (idx < 0 || idx >= nAlleles)
          throw new ArrayIndexOutOfBoundsException(idx)

        (ploidy(c): @switch) match {
          case 0 => 0
          case 1 => (alleleRepr(c) == idx).toInt
          case 2 =>
            var r = 0
            if (idx == j)
              r += 1
            if (idx == k)
              r += 1
            r
          case _ => throw new UnsupportedOperationException
        }
      }
    }
  }

  def oneHotAlleles(c: Call, v: Variant): IndexedSeq[Int] = oneHotAlleles(c, v.nAlleles)

  def check(c: Call, nAlleles: Int) {
    (ploidy(c): @switch) match {
      case 0 =>
      case 1 =>
        val a = alleleByIndex(c, 0)
        assert(a >= 0 && a < nAlleles)
      case 2 =>
        val nGenotypes = triangle(nAlleles)
        val udtn =
          if (isPhased(c)) {
            val p = allelePair(c)
            unphasedDiploidGtIndex(Call2(p.j, p.k))
          } else
            unphasedDiploidGtIndex(c)
        assert(udtn < nGenotypes, s"Invalid call found `${ c.toString }' for number of alleles equal to `$nAlleles'.")
      case _ =>
        alleles(c).foreach(a => assert(a >= 0 && a < nAlleles))
    }
  }

  def gen(nAlleles: Int, ploidyGen: Gen[Int] = Gen.choose(0, 2), phasedGen: Gen[Boolean] = Gen.nextCoin(0.5)): Gen[Call] = for {
    ploidy <- ploidyGen
    phased <- phasedGen
    alleles <- Gen.buildableOfN[Array](ploidy, Gen.choose(0, nAlleles - 1))
  } yield {
    val c = CallN(alleles, phased)
    check(c, nAlleles)
    c
  }

  def genUnphasedDiploid(nAlleles: Int): Gen[Call] = gen(nAlleles, Gen.const(2), Gen.const(false))

  def genPhasedDiploid(nAlleles: Int): Gen[Call] = gen(nAlleles, Gen.const(2), Gen.const(true))

  def genNonmissingValue: Gen[Call] = for {
    nAlleles <- Gen.choose(2, 5)
    c <- gen(nAlleles)
  } yield {
    check(c, nAlleles)
    c
  }
}
