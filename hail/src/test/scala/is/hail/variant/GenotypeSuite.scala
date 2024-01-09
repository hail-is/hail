package is.hail.variant

import is.hail.TestUtils
import is.hail.check.Gen
import is.hail.check.Prop._
import is.hail.testUtils.Variant
import is.hail.utils._

import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

class GenotypeSuite extends TestNGSuite {

  val v = Variant("1", 1, "A", "T")

  @Test def gtPairGtIndexIsId() {
    forAll(Gen.choose(0, 32768), Gen.choose(0, 32768)) { (x, y) =>
      val (j, k) = if (x < y) (x, y) else (y, x)
      val gt = AllelePair(j, k)
      Genotype.allelePair(Genotype.diploidGtIndex(gt)) == gt
    }.check()
  }

  def triangleNumberOf(i: Int) = (i * i + i) / 2

  @Test def gtIndexGtPairIsId() {
    forAll(Gen.choose(0, 10000)) { (idx) =>
      Genotype.diploidGtIndex(Genotype.allelePair(idx)) == idx
    }.check()
  }

  @Test def gtPairAndGtPairSqrtEqual() {
    forAll(Gen.choose(0, 10000)) { (idx) =>
      Genotype.allelePair(idx) == Genotype.allelePairSqrt(idx)
    }.check()
  }

  @Test def testGtFromLinear() {
    val gen = for {
      nGenotype <- Gen.choose(2, 5).map(triangleNumberOf)
      dosageGen = Gen.partition(nGenotype, 32768)
      result <- dosageGen
    } yield result

    val p = forAll(gen) { gp =>
      val gt = Option(uniqueMaxIndex(gp))
      assert(gp.sum == 32768)
      val dMax = gp.max

      val check1 = gt.forall { gt =>
        val dosageP = gp(gt)
        dosageP == dMax && gp.zipWithIndex.forall { case (d, index) => index == gt || d != dosageP }
      }

      val check2 = gp.count(_ == dMax) > 1 || gt.contains(gp.indexOf(dMax))

      check1 && check2
    }
    p.check()
  }

  @Test def testPlToDosage() {
    val gt0 = Genotype.plToDosage(0, 20, 100)
    val gt1 = Genotype.plToDosage(20, 0, 100)
    val gt2 = Genotype.plToDosage(20, 100, 0)

    assert(D_==(gt0, 0.009900990296049406))
    assert(D_==(gt1, 0.9900990100009803))
    assert(D_==(gt2, 1.980198019704931))
  }

  @Test def testCall() {
    assert((0 until 9).forall { gt =>
      val c = Call2.fromUnphasedDiploidGtIndex(gt)
      !Call.isPhased(c) &&
      Call.ploidy(c) == 2 &&
      Call.isDiploid(c) &&
      Call.isUnphasedDiploid(c) &&
      Call.unphasedDiploidGtIndex(c) == gt &&
      Call.alleleRepr(c) == gt
    })

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
      Call.ploidy(c) == 2
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
    intercept[UnsupportedOperationException](Call.parse("1/1/1"))
    intercept[UnsupportedOperationException](Call.parse("1|1|1"))
    TestUtils.interceptFatal("invalid call expression:")(Call.parse("0/"))
  }
}
