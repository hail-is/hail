package is.hail.variant

import is.hail.check.Prop._
import is.hail.check.{Gen, Properties}
import is.hail.utils.{ByteIterator, _}
import org.scalatest.testng.TestNGSuite
import org.testng.annotations.Test

import scala.collection.mutable

object GenotypeSuite {
  val ab = new ArrayBuilder[Byte]()

  def readWriteEqual(nAlleles: Int, g: Genotype): Boolean = {
    ab.clear()

    val isLinearScale = g.isLinearScale
    val gb = new GenotypeBuilder(nAlleles, isLinearScale)

    gb.set(g)
    gb.write(ab)
    val g2 = Genotype.read(nAlleles, isLinearScale, new ByteIterator(ab.result()))

    g == g2
  }

  object Spec extends Properties("Genotype") {

    property("readWrite") = forAll[(Variant, Genotype)](Genotype.genVariantGenotype) { case (v, g) =>
      readWriteEqual(v.nAlleles, g)
    }

    property("gt") = forAll { g: Genotype =>
      g.gt.isDefined == g.isCalled
    }

    property("gtPairIndex") = forAll(Gen.choose(0, 0x7fff),
      Gen.choose(0, 0x7fff)) { (i: Int, j: Int) =>
      (i <= j) ==> (Genotype.gtPair(Genotype.gtIndex(i, j)) == GTPair(i, j))
    }

    property("gtIndexPair") = forAll(Gen.choose(0, 0x20003fff)) { (i: Int) =>
      val p = Genotype.gtPair(i)

      Genotype.gtIndex(p) == i &&
        Genotype.gtPairSqrt(i) == p &&
        Genotype.gtPairRecursive(i) == p
    }
  }

}

class GenotypeSuite extends TestNGSuite {

  import GenotypeSuite._

  val v = Variant("1", 1, "A", "T")

  def testReadWrite(g: Genotype) {
    assert(readWriteEqual(v.nAlleles, g))
  }

  @Test def testGenotype() {
    intercept[IllegalArgumentException] {
      Genotype(Some(-2), Some(Array(2, 0)), Some(2), None)
    }

    val noCall = Genotype(None, Some(Array(2, 0)), Some(2), None)
    val homRef = Genotype(Some(0), Some(Array(10, 0)), Some(10), Some(99), Some(Array(0, 1000, 100)))
    val het = Genotype(Some(1), Some(Array(5, 5)), Some(12), Some(99), Some(Array(100, 0, 1000)))
    val homVar = Genotype(Some(2), Some(Array(2, 10)), Some(12), Some(99), Some(Array(100, 1000, 0)))

    assert(noCall.isNotCalled && !noCall.isCalled && !noCall.isHomRef && !noCall.isHet && !noCall.isHomVar)
    assert(!homRef.isNotCalled && homRef.isCalled && homRef.isHomRef && !homRef.isHet && !homRef.isHomVar)
    assert(!het.isNotCalled && het.isCalled && !het.isHomRef && het.isHet && !het.isHomVar)
    assert(!homVar.isNotCalled && homVar.isCalled && !homVar.isHomRef && !homVar.isHet && homVar.isHomVar)

    assert(noCall.gt.isEmpty)
    assert(homRef.gt.isDefined)
    assert(het.gt.isDefined)
    assert(homVar.gt.isDefined)

    testReadWrite(noCall)
    testReadWrite(homRef)
    testReadWrite(het)
    testReadWrite(homVar)

    assert(Genotype(None, None, None, None).pAB().isEmpty)
    assert(Genotype(None, Some(Array(0, 0)), Some(0), None, None).pAB().isEmpty)
    assert(D_==(Genotype(Some(1), Some(Array(16, 16)), Some(33), Some(99), Some(Array(100, 0, 100))).pAB().get, 1.0))
    assert(D_==(Genotype(Some(4), Some(Array(16, 16, 16)), Some(48), None, None).pAB().get, 1.0))
    assert(D_==(Genotype(Some(4), Some(Array(16, 5, 8)), Some(48), None, None).pAB().get, 0.423950))
    assert(D_==(Genotype(Some(1), Some(Array(5, 8)), Some(13), Some(99), Some(Array(200, 0, 100))).pAB().get, 0.423950))

    Spec.check()
  }

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

  @Test def testUnboxedDosage() {
    val g0 = Genotype.apply(gt = Some(0), px = Some(Array(32648, 20, 100)), isLinearScale = true)
    val g1 = Genotype.apply(gt = Some(1), px = Some(Array(20, 32648, 100)), isLinearScale = true)
    val g2 = Genotype.apply(gt = Some(2), px = Some(Array(20, 100, 32648)), isLinearScale = true)

    assert(D_==(g0.unboxedDosage, (20 + 2 * 100) / 32768d))
    assert(D_==(g1.unboxedDosage, (32648 + 2 * 100) / 32768d))
    assert(D_==(g2.unboxedDosage, (100 + 2 * 32648) / 32768d))
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
