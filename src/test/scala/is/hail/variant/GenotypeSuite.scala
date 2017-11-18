package is.hail.variant

import is.hail.check.Prop._
import is.hail.check.{Gen, Properties}
import is.hail.utils.{ByteIterator, _}
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

import scala.collection.mutable

class GenotypeSuite extends TestNGSuite {

  val v = Variant("1", 1, "A", "T")

  @Test def gtPairGtIndexIsId() {
    forAll(Gen.choose(0, 32768), Gen.choose(0, 32768)) { (x, y) =>
      val (j, k) = if (x < y) (x, y) else (y, x)
      val gt = GTPair(j, k)
      Genotype.gtPair(Genotype.gtIndex(gt)) == gt
    }.check()
  }

  def triangleNumberOf(i: Int) = (i * i + i) / 2

  @Test def gtIndexGtPairIsId() {
    forAll(Gen.choose(0, 10000)) { (idx) =>
      Genotype.gtIndex(Genotype.gtPair(idx)) == idx
    }.check()
  }

  @Test def gtPairAndGtPairSqrtEqual() {
    forAll(Gen.choose(0, 10000)) { (idx) =>
      Genotype.gtPair(idx) == Genotype.gtPairSqrt(idx)
    }.check()
  }

  @Test def testGtFromLinear() {
    val gen = for (nGenotype <- Gen.choose(2, 5).map(triangleNumberOf);
      dosageGen = Gen.partition(nGenotype, 32768);
      result <- dosageGen) yield result

    val p = forAll(gen) { gp =>
      val gt = Genotype.gtFromLinear(gp)
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
}
