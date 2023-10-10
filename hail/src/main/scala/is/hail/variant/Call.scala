package is.hail.variant

import is.hail.check.Gen
import is.hail.expr.Parser
import is.hail.utils._

import java.io.Serializable
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
    if ((gt >>> 29) != 0)
      fatal(s"invalid allele representation: $gt. Max value is 2^29 - 1", -1)
    val ploidy = 2
    ploidy << 1 | gt << 3
  }

  def apply(aj: Int, ak: Int, phased: Boolean = false): Call = {
    if (aj < 0 || ak < 0)
      fatal(s"allele indices must be >= 0. Found j=$aj and k=$ak.")

    if (phased) {
      Call(Genotype.diploidGtIndex(aj, aj + ak), true, ploidy = 2)
    } else {
      fromUnphasedDiploidGtIndex(Genotype.diploidGtIndexWithSwap(aj, ak))
    }
  }

  def withErrorID(aj: Int, ak: Int, phased: Boolean, errorID: Int): Call = {
    if (aj < 0 || ak < 0)
      fatal(s"allele indices must be >= 0. Found j=$aj and k=$ak.", errorID)

    val alleleRepr =
      if (phased)
        Genotype.diploidGtIndex(aj, aj + ak)
      else
        Genotype.diploidGtIndexWithSwap(aj, ak)

    Call(alleleRepr, phased, ploidy = 2, errorID = errorID)
  }
}

object CallN {
  def apply(alleles: java.util.List[Int], phased: Boolean): Call = apply(alleles.asScala.toFastSeq, phased)

  def apply(alleles: IndexedSeq[Int], phased: Boolean = false): Call = {
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
  def apply(ar: Int, phased: Boolean, ploidy: Int, errorID: Int = -1): Call = {
    if (ploidy < 0 || ploidy > 2)
      fatal(s"invalid ploidy: $ploidy. Only support ploidy in range [0, 2]", errorID)
    if (ar < 0)
      fatal(s"invalid allele representation: $ar. Must be positive.", errorID)

    var c = 0
    c |= phased.toInt

    if (ploidy > 2)
      c |= (3 << 1)
    else
      c |= (ploidy << 1)

    if ((ar >>> 29) != 0)
      fatal(s"invalid allele representation: $ar. Max value is 2^29 - 1", errorID)

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

  def allelePair(c: Call): Int = {
    if (!isDiploid(c))
      fatal(s"invalid ploidy: ${ ploidy(c) }. Only support ploidy == 2")
    allelePairUnchecked(c)
  }

  def allelePairUnchecked(c: Call): Int = {
    if (isPhased(c)) {
      val p = Genotype.allelePair(alleleRepr(c))
      val j = AllelePair.j(p)
      val k = AllelePair.k(p)
      AllelePair(j, k - j)
    } else
      Genotype.allelePair(alleleRepr(c))
  }

  def unphasedDiploidGtIndex(c: Call): Int = {
    if (!isDiploid(c))
      fatal(s"unphased_diploid_gt_index only supports ploidy == 2. Found ${ Call.toString(c) }.")
    if (isPhased(c)) {
      val p = Genotype.allelePair(alleleRepr(c))
      val j = AllelePair.j(p)
      val k = AllelePair.k(p)
      Genotype.diploidGtIndexWithSwap(j, k - j)
    } else
      alleleRepr(c)
  }

  def alleles(c: Call): Array[Int] = {
    (ploidy(c): @switch) match {
      case 0 => Array.empty[Int]
      case 1 => Array(alleleByIndex(c, 0))
      case 2 => AllelePair.alleleIndices(allelePair(c))
      case _ => throw new UnsupportedOperationException
    }
  }

  def alleleByIndex(c: Call, i: Int): Int = {
    (ploidy(c): @switch) match {
      case 0 => throw new UnsupportedOperationException
      case 1 =>
        if (i != 0)
          fatal(s"Index out of bounds for call with ploidy=1: $i")
        alleleRepr(c)
      case 2 =>
        if (i != 0 && i != 1)
          fatal(s"Index out of bounds for call with ploidy=2: $i")
        val p = allelePair(c)
        if (i == 0) AllelePair.j(p) else AllelePair.k(p)
      case _ =>
        if (i < 0 || i >= ploidy(c))
          fatal(s"Index out of bounds for call with ploidy=${ ploidy(c) }: $i")
        alleles(c)(i)
    }
  }

  def downcode(c: Call, i: Int): Call = {
    (Call.ploidy(c): @switch) match {
      case 0 => c
      case 1 =>
        Call1(if (Call.alleleByIndex(c, 0) == i) 1 else 0, Call.isPhased(c))
      case 2 =>
        val p = Call.allelePair(c)
        Call2(if (AllelePair.j(p) == i) 1 else 0, if (AllelePair.k(p) == i) 1 else 0, Call.isPhased(c))
      case _ =>
        CallN(Call.alleles(c).map(a => if (a == i) 1 else 0), Call.isPhased(c))
    }
  }

  def unphase(c: Call): Call = {
    (Call.ploidy(c): @switch) match {
      case 0 => c
      case 1 => Call1(Call.alleleByIndex(c, 0))
      case 2 =>
        val p = allelePair(c)
        val j = AllelePair.j(p)
        val k = AllelePair.k(p)
        if (j <= k) Call2(j, k) else Call2(k, j)
    }
  }

  def containsAllele(c: Call, allele: Int): Boolean = {
    (Call.ploidy(c): @switch) match {
      case 0 => false
      case 1 => Call.alleleByIndex(c, 0) == allele
      case 2 =>
        val p = allelePair(c)
        AllelePair.j(p) == allele || AllelePair.k(p) == allele
    }
  }

  def parse(s: String): Call = Parser.parseCall(s)

  def toString(c: Call): String = {
    val phased = isPhased(c)
    if (phased) {
      (ploidy(c): @switch) match {
        case 0 => "|-"
        case 1 =>
          val a = alleleByIndex(c, 0)

          (a: @switch) match {
            case 0 => "|0"
            case 1 => "|1"
            case 2 => "|2"
            case _ => s"|$a"
          }
        case 2 =>
          (alleleRepr(c): @switch) match {
            case 0 => "0|0"
            case 1 => "0|1"
            case 2 => "1|0"
            case 3 => "0|2"
            case 4 => "1|1"
            case _ =>
              val p = allelePair(c)
              s"${ AllelePair.j(p) }|${ AllelePair.k(p) }"
          }
        case _ =>
          alleles(c).mkString("|")
      }
    } else {
      (ploidy(c): @switch) match {
        case 0 => "-"
        case 1 =>
          val a = alleleByIndex(c, 0)

          (a: @switch) match {
            case 0 => "0"
            case 1 => "1"
            case 2 => "2"
            case _ => a.toString
          }
        case 2 =>
          (alleleRepr(c): @switch) match {
            case 0 => "0/0"
            case 1 => "0/1"
            case 2 => "1/1"
            case _ =>
              val p = allelePair(c)
              s"${ AllelePair.j(p) }/${ AllelePair.k(p) }"
          }
        case _ =>
          alleles(c).mkString("/")
      }
    }
  }

  private[this] val phased_ = Array[Byte]('|', '-')
  private[this] val phased_0 = Array[Byte]('|', '0')
  private[this] val phased_1 = Array[Byte]('|', '1')
  private[this] val phased_2 = Array[Byte]('|', '2')
  private[this] val phased_00 = Array[Byte]('0', '|', '0')
  private[this] val phased_01 = Array[Byte]('0', '|', '1')
  private[this] val phased_10 = Array[Byte]('1', '|', '0')
  private[this] val phased_02 = Array[Byte]('0', '|', '2')
  private[this] val phased_11 = Array[Byte]('1', '|', '1')

  private[this] val unphased_ = Array[Byte]('-')
  private[this] val unphased_0 = Array[Byte]('0')
  private[this] val unphased_1 = Array[Byte]('1')
  private[this] val unphased_2 = Array[Byte]('2')
  private[this] val unphased_00 = Array[Byte]('0', '/', '0')
  private[this] val unphased_01 = Array[Byte]('0', '/', '1')
  private[this] val unphased_11 = Array[Byte]('1', '/', '1')

  def toUTF8(c: Call): Array[Byte] = {
    val phased = isPhased(c)
    if (phased) {
      (ploidy(c): @switch) match {
        case 0 => phased_
        case 1 =>
          val a = alleleByIndex(c, 0)

          (a: @switch) match {
            case 0 => phased_0
            case 1 => phased_1
            case 2 => phased_2
            case _ => s"|$a".getBytes()
          }
        case 2 =>
          (alleleRepr(c): @switch) match {
            case 0 => phased_00
            case 1 => phased_01
            case 2 => phased_10
            case 3 => phased_02
            case 4 => phased_11
            case _ =>
              val p = allelePair(c)
              s"${ AllelePair.j(p) }|${ AllelePair.k(p) }".getBytes()
          }
        case _ =>
          alleles(c).mkString("|").getBytes()
      }
    } else {
      (ploidy(c): @switch) match {
        case 0 => unphased_
        case 1 =>
          val a = alleleByIndex(c, 0)

          (a: @switch) match {
            case 0 => unphased_0
            case 1 => unphased_1
            case 2 => unphased_2
            case _ => a.toString.getBytes()
          }
        case 2 =>
          (alleleRepr(c): @switch) match {
            case 0 => unphased_00
            case 1 => unphased_01
            case 2 => unphased_11
            case _ =>
              val p = allelePair(c)
              s"${ AllelePair.j(p) }/${ AllelePair.k(p) }".getBytes()
          }
        case _ =>
          alleles(c).mkString("/").getBytes()
      }
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
        sb.append(AllelePair.j(p))
        sb.append(sep)
        sb.append(AllelePair.k(p))
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
      case 1 => alleleRepr(c) == 0
      case 2 => alleleRepr(c) == 0
      case _ => alleles(c).forall(_ == 0)
    }
  }

  def isHet(c: Call): Boolean = {
    (ploidy(c): @switch) match {
      case 0 | 1 => false
      case 2 => alleleRepr(c) > 0 && {
        val p = allelePairUnchecked(c)
        AllelePair.j(p) != AllelePair.k(p)
      }
      case _ => throw new UnsupportedOperationException
    }
  }

  def isHomVar(c: Call): Boolean = {
    (ploidy(c): @switch) match {
      case 0 => false
      case 1 => alleleRepr(c) > 0
      case 2 => alleleRepr(c) > 0 && {
        val p = allelePairUnchecked(c)
        AllelePair.j(p) == AllelePair.k(p)
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
        val p = allelePairUnchecked(c)
        val j = AllelePair.j(p)
        val k = AllelePair.k(p)
        j > 0 && k > 0 && k != j
      }
      case _ => throw new UnsupportedOperationException
    }
  }

  def isHetRef(c: Call): Boolean = {
    (ploidy(c): @switch) match {
      case 0 | 1 => false
      case 2 => alleleRepr(c) > 0 && {
        val p = allelePairUnchecked(c)
        val j = AllelePair.j(p)
        val k = AllelePair.k(p)
        (j == 0 && k > 0) || (k == 0 && j > 0)
      }
      case _ => throw new UnsupportedOperationException
    }
  }

  def nNonRefAlleles(c: Call): Int = {
    (ploidy(c): @switch) match {
      case 0 => 0
      case 1 => (alleleRepr(c) > 0).toInt
      case 2 => AllelePair.nNonRefAlleles(allelePair(c))
      case _ => alleles(c).count(_ != 0)
    }
  }

  def oneHotAlleles(c: Call, nAlleles: Int): IndexedSeq[Int] = {
    var j = 0
    var k = 0

    if (ploidy(c) == 2) {
      val p = allelePair(c)
      j = AllelePair.j(p)
      k = AllelePair.k(p)
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
            unphasedDiploidGtIndex(Call2(AllelePair.j(p), AllelePair.k(p)))
          } else
            unphasedDiploidGtIndex(c)
        assert(udtn < nGenotypes, s"Invalid call found '${ c.toString }' for number of alleles equal to '$nAlleles'.")
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
