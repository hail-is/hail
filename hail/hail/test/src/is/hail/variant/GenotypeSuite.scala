package is.hail.variant

import is.hail.scalacheck.partition
import is.hail.testUtils.Variant
import is.hail.utils._

import org.scalacheck.Gen
import org.scalatest
import org.scalatestplus.scalacheck.CheckerAsserting.assertingNatureOfAssertion
import org.scalatestplus.scalacheck.ScalaCheckDrivenPropertyChecks
import org.scalatestplus.testng.TestNGSuite
import org.testng.annotations.Test

class GenotypeSuite extends TestNGSuite with ScalaCheckDrivenPropertyChecks {

  val v = Variant("1", 1, "A", "T")

  @Test def gtPairGtIndexIsId(): Unit =
    forAll(Gen.choose(0, 32768), Gen.choose(0, 32768)) { (x, y) =>
      val (j, k) = if (x < y) (x, y) else (y, x)
      val gt = AllelePair(j, k)
      assert(Genotype.allelePair(Genotype.diploidGtIndex(gt)) == gt)
    }

  def triangleNumberOf(i: Int): Int =
    (i * i + i) / 2

  @Test def gtIndexGtPairIsId(): Unit =
    forAll(Gen.choose(0, 10000)) { idx =>
      assert(Genotype.diploidGtIndex(Genotype.allelePair(idx)) == idx)
    }

  @Test def gtPairAndGtPairSqrtEqual(): Unit =
    forAll(Gen.choose(0, 10000)) { idx =>
      assert(Genotype.allelePair(idx) == Genotype.allelePairSqrt(idx))
    }

  @Test def testGtFromLinear(): Unit = {
    val gen =
      for {
        nGenotype <- Gen.choose(2, 5).map(triangleNumberOf)
        result <- partition(nGenotype, 32768)
      } yield result

    forAll(gen) { gp =>
      val gt = Option(uniqueMaxIndex(gp))
      assert(gp.sum == 32768)
      val dMax = gp.max

      scalatest.Inspectors.forAll(gt.toSeq) { gt =>
        val dosageP = gp(gt)
        dosageP == dMax && gp.zipWithIndex.forall { case (d, index) => index == gt || d != dosageP }
      }

      assert(gp.count(_ == dMax) > 1 || gt.contains(gp.indexOf(dMax)))
    }
  }

  @Test def testPlToDosage(): Unit = {
    val gt0 = Genotype.plToDosage(0, 20, 100)
    val gt1 = Genotype.plToDosage(20, 0, 100)
    val gt2 = Genotype.plToDosage(20, 100, 0)

    assert(D_==(gt0, 0.009900990296049406))
    assert(D_==(gt1, 0.9900990100009803))
    assert(D_==(gt2, 1.980198019704931))
  }

  @Test def testCall(): Unit = {
    scalatest.Inspectors.forAll(0 until 9) { gt =>
      val c = Call2.fromUnphasedDiploidGtIndex(gt)
      assert(
        !Call.isPhased(c) &&
          Call.ploidy(c) == 2 &&
          Call.isDiploid(c) &&
          Call.isUnphasedDiploid(c) &&
          Call.unphasedDiploidGtIndex(c) == gt &&
          Call.alleleRepr(c) == gt
      )
    }

    val c0 = Call2(0, 0, phased = true)
    val c1a = Call2(0, 1, phased = true)
    val c1b = Call2(1, 0, phased = true)
    val c2 = Call2(1, 1, phased = true)
    val c4 = Call2(2, 1, phased = true)

    val x = Array((c0, 0, 0), (c1a, 1, 1), (c1b, 1, 2), (c2, 2, 4), (c4, 4, 8))

    assert(x.forall { case (c, unphasedGt, alleleRepr) =>
      val alleles = Call.alleles(c)
      c != Call2.fromUnphasedDiploidGtIndex(unphasedGt) &&
      Call.isPhased(c) &&
      Call.ploidy(c) == 2 &&
      Call.isDiploid(c) &&
      !Call.isUnphasedDiploid(c) &&
      Call.unphasedDiploidGtIndex(Call2(alleles(0), alleles(1))) == unphasedGt &&
      Call.alleleRepr(c) == alleleRepr
    })

    assert(Call.isHomRef(c0) && !Call.isHet(c0) && !Call.isHomVar(c0) &&
      !Call.isHetNonRef(c0) && !Call.isHetRef(c0) && !Call.isNonRef(c0))

    assert(!Call.isHomRef(c1a) && Call.isHet(c1a) && !Call.isHomVar(c1a) &&
      !Call.isHetNonRef(c1a) && Call.isHetRef(c1a) && Call.isNonRef(c1a))

    assert(!Call.isHomRef(c1b) && Call.isHet(c1b) && !Call.isHomVar(c1b) &&
      !Call.isHetNonRef(c1b) && Call.isHetRef(c1b) && Call.isNonRef(c1b))

    assert(!Call.isHomRef(c2) && !Call.isHet(c2) && Call.isHomVar(c2) &&
      !Call.isHetNonRef(c2) && !Call.isHetRef(c2) && Call.isNonRef(c2))

    assert(!Call.isHomRef(c4) && Call.isHet(c4) && !Call.isHomVar(c4) &&
      Call.isHetNonRef(c4) && !Call.isHetRef(c4) && Call.isNonRef(c4))

    assert(Call.parse("-") == Call0())
    assert(Call.parse("|-") == Call0(true))
    assert(Call.parse("1") == Call1(1))
    assert(Call.parse("|1") == Call1(1, phased = true))
    assert(Call.parse("0/0") == Call2(0, 0))
    assert(Call.parse("0|1") == Call2(0, 1, phased = true))
    assertThrows[UnsupportedOperationException](Call.parse("1/1/1"))
    assertThrows[UnsupportedOperationException](Call.parse("1|1|1"))
    val he = intercept[HailException](Call.parse("0/"))
    assert(he.msg.contains("invalid call expression"))
  }
}
