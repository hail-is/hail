package is.hail.variant

import is.hail.scalacheck.partition
import is.hail.utils._

import org.scalacheck.Gen
import org.scalacheck.Prop.forAll

class GenotypeSuite extends munit.ScalaCheckSuite {

  property("gtPairGtIndexIsId") =
    forAll(Gen.choose(0, 32768), Gen.choose(0, 32768)) { (x, y) =>
      val (j, k) = if (x < y) (x, y) else (y, x)
      val gt = AllelePair(j, k)

      Genotype.allelePair(Genotype.diploidGtIndex(gt)) == gt
    }

  def triangleNumberOf(i: Int): Int =
    (i * i + i) / 2

  property("gtIndexGtPairIsId") =
    forAll(Gen.choose(0, 10000))(idx => Genotype.diploidGtIndex(Genotype.allelePair(idx)) == idx)

  property("gtPairAndGtPairSqrtEqual") =
    forAll(Gen.choose(0, 10000))(idx => Genotype.allelePair(idx) == Genotype.allelePairSqrt(idx))

  property("GtFromLinear") = {
    val gen =
      for {
        nGenotype <- Gen.choose(2, 5).map(triangleNumberOf)
        result <- partition(nGenotype, 32768)
      } yield result

    forAll(gen) { gp =>
      val gt = Option(uniqueMaxIndex(gp))
      assertEquals(gp.sum, 32768)
      val dMax = gp.max

      gt.toSeq.foreach { gt =>
        val dosageP = gp(gt)
        dosageP == dMax && gp.zipWithIndex.forall { case (d, index) => index == gt || d != dosageP }
      }

      gp.count(_ == dMax) > 1 || gt.contains(gp.indexOf(dMax))
    }
  }

  test("PlToDosage") {
    val gt0 = Genotype.plToDosage(0, 20, 100)
    val gt1 = Genotype.plToDosage(20, 0, 100)
    val gt2 = Genotype.plToDosage(20, 100, 0)

    assert(D_==(gt0, 0.009900990296049406))
    assert(D_==(gt1, 0.9900990100009803))
    assert(D_==(gt2, 1.980198019704931))
  }

  test("Call") {
    (0 until 9).foreach { gt =>
      val c = Call2.fromUnphasedDiploidGtIndex(gt)
      assert(!Call.isPhased(c))
      assertEquals(Call.ploidy(c), 2)
      assert(Call.isDiploid(c))
      assert(Call.isUnphasedDiploid(c))
      assertEquals(Call.unphasedDiploidGtIndex(c), gt)
      assertEquals(Call.alleleRepr(c), gt)
    }

    val c0 = Call2(0, 0, phased = true)
    val c1a = Call2(0, 1, phased = true)
    val c1b = Call2(1, 0, phased = true)
    val c2 = Call2(1, 1, phased = true)
    val c4 = Call2(2, 1, phased = true)

    val x = Array((c0, 0, 0), (c1a, 1, 1), (c1b, 1, 2), (c2, 2, 4), (c4, 4, 8))

    x.foreach { case (c, unphasedGt, alleleRepr) =>
      val alleles = Call.alleles(c)
      assert(c != Call2.fromUnphasedDiploidGtIndex(unphasedGt))
      assert(Call.isPhased(c))
      assertEquals(Call.ploidy(c), 2)
      assert(Call.isDiploid(c))
      assert(!Call.isUnphasedDiploid(c))
      assertEquals(Call.unphasedDiploidGtIndex(Call2(alleles(0), alleles(1))), unphasedGt)
      assertEquals(Call.alleleRepr(c), alleleRepr)
    }

    assert(Call.isHomRef(c0))
    assert(!Call.isHet(c0))
    assert(!Call.isHomVar(c0))
    assert(!Call.isHetNonRef(c0))
    assert(!Call.isHetRef(c0))
    assert(!Call.isNonRef(c0))

    assert(!Call.isHomRef(c1a))
    assert(Call.isHet(c1a))
    assert(!Call.isHomVar(c1a))
    assert(!Call.isHetNonRef(c1a))
    assert(Call.isHetRef(c1a))
    assert(Call.isNonRef(c1a))

    assert(!Call.isHomRef(c1b))
    assert(Call.isHet(c1b))
    assert(!Call.isHomVar(c1b))
    assert(!Call.isHetNonRef(c1b))
    assert(Call.isHetRef(c1b))
    assert(Call.isNonRef(c1b))

    assert(!Call.isHomRef(c2))
    assert(!Call.isHet(c2))
    assert(Call.isHomVar(c2))
    assert(!Call.isHetNonRef(c2))
    assert(!Call.isHetRef(c2))
    assert(Call.isNonRef(c2))

    assert(!Call.isHomRef(c4))
    assert(Call.isHet(c4))
    assert(!Call.isHomVar(c4))
    assert(Call.isHetNonRef(c4))
    assert(!Call.isHetRef(c4))
    assert(Call.isNonRef(c4))

    assertEquals(Call.parse("-"), Call0())
    assertEquals(Call.parse("|-"), Call0(true))
    assertEquals(Call.parse("1"), Call1(1))
    assertEquals(Call.parse("|1"), Call1(1, phased = true))
    assertEquals(Call.parse("0/0"), Call2(0, 0))
    assertEquals(Call.parse("0|1"), Call2(0, 1, phased = true))
    intercept[UnsupportedOperationException](Call.parse("1/1/1")): Unit
    intercept[UnsupportedOperationException](Call.parse("1|1|1")): Unit
    val he = intercept[HailException](Call.parse("0/"))
    assert(he.msg.contains("invalid call expression"))
  }
}
