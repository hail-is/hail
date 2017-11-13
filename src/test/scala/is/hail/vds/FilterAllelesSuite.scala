package is.hail.vds

import is.hail.SparkSuite
import is.hail.annotations._
import is.hail.check.Prop
import is.hail.expr.{TString, TStruct}
import is.hail.utils._
import is.hail.variant.{AltAllele, Genotype, VSMFileMetadata, VSMSubgen, Variant, VariantSampleMatrix}
import org.testng.annotations.Test

class FilterAllelesSuite extends SparkSuite {

  @Test def testRandom() {
    Prop.forAll(VSMSubgen.random.gen(hc)) { vds =>
      val vds2 = vds.annotateAllelesExpr("va.p = pcoin(0.2)")
        .cache()

      val (nAlleles1, _) = vds2.queryVariants("variants.map(v => va.p.map(x => if (x) 1 else 0).sum()).sum()")

      val nAlleles2 = vds2.filterAlleles("va.p[aIndex - 1]", keepStar = true)
        .splitMulti(keepStar = true)
        .countVariants()

      assert(nAlleles1 == nAlleles2)

      true
    }.check()
  }

  @Test def filterAllAlleles(): Unit = {
    Prop.forAll(VSMSubgen.random.gen(hc)) { vds =>
      vds.filterAlleles("false", subset = true).countVariants() == 0
    }.check()
  }

  @Test def filterNoAlleles(): Unit = {
    Prop.forAll(VSMSubgen.random.gen(hc)) { vds =>
      vds.filterAlleles("true", subset = true, keepStar = true)
        .countVariants() == vds.countVariants()
    }.check()
  }

  @Test def filterSecondOfTwoAllelesDowncode(): Unit = {
    val variant1 = new Variant("contig", 0, "ref", IndexedSeq("alt1", "alt2").map(AltAllele("ref", _)))
    val va1 = null
    val genotype11 = Genotype(Option(1), Option(Array(25, 5, 30)), Option(100), Option(5), Option(Array(10, 0, 10, 10, 5, 7)))
    val genotype12 = Genotype(Option(2), Option(Array(25, 35, 0)), Option(100), Option(5), Option(Array(10, 10, 0, 7, 5, 10)))
    val genotype13 = Genotype(Option(3), Option(Array(25, 10, 10)), Option(100), Option(5), Option(Array(10, 10, 10, 0, 5, 7)))
    val genotypes1 = Seq(genotype11, genotype12, genotype13)
    val row1: (Variant, (Annotation, Iterable[Genotype])) = (variant1, (va1, genotypes1))

    val vds = new VariantSampleMatrix(hc, VSMFileMetadata(Array("1", "2", "3"),
      IndexedSeq[Annotation](null, null, null),
      null,
      TString(),
      TStruct.empty),
      sc.parallelize(Seq(row1)).toOrderedRDD)
      .filterAlleles("aIndex == 2", subset = false, keep = false)

    val newVariant1 = variant1.copy(altAlleles = variant1.altAlleles.take(1))
    val newVa1 = va1: Annotation
    val newGenotype11 = genotype11.copy(ad = Option(Array(55, 5)), gq = Option(7), px = Option(Array(7, 0, 10)))
    val newGenotype12 = genotype12.copy(ad = Option(Array(25, 35)), px = Option(Array(7, 5, 0)))
    val newGenotype13 = genotype13.copy(gt = Option(0), ad = Option(Array(35, 10)), px = Option(Array(0, 5, 10)))
    val newGenotypes1 = Seq(newGenotype11, newGenotype12, newGenotype13)
    val newRow1 = (newVariant1, (newVa1, newGenotypes1))

    assert(vds.rdd.collect().toSeq == Seq(newRow1))
  }

  @Test def filterFirstOfTwoAllelesDowncode(): Unit = {
    val variant1 = new Variant("contig", 0, "ref", IndexedSeq("alt1", "alt2").map(AltAllele("ref", _)))
    val va1 = null
    val genotype11 = Genotype(Option(1), Option(Array(25, 5, 30)), Option(100), Option(5), Option(Array(10, 0, 10, 10, 5, 7)))
    val genotype12 = Genotype(Option(2), Option(Array(25, 35, 0)), Option(100), Option(5), Option(Array(10, 10, 0, 7, 5, 10)))
    val genotype13 = Genotype(Option(3), Option(Array(25, 10, 10)), Option(100), Option(5), Option(Array(10, 10, 10, 0, 5, 7)))
    val genotypes1 = Seq(genotype11, genotype12, genotype13)
    val row1: (Variant, (Annotation, Iterable[Genotype])) = (variant1, (va1, genotypes1))

    val vds = new VariantSampleMatrix(hc, VSMFileMetadata(Array("1", "2", "3"),
      IndexedSeq[Annotation](null, null, null),
      null,
      TString(),
      TStruct.empty),
      sc.parallelize(Seq(row1)).toOrderedRDD)
      .filterAlleles("aIndex == 1", subset = false, keep = false)

    val newVariant1 = variant1.copy(altAlleles = variant1.altAlleles.drop(1))
    val newVa1 = va1
    val newGenotype11 = genotype11.copy(gt = Option(0), ad = Option(Array(30, 30)), px = Option(Array(0, 5, 7)))
    val newGenotype12 = genotype12.copy(gt = Option(0), ad = Option(Array(60, 0)), px = Option(Array(0, 5, 10)))
    val newGenotype13 = genotype13.copy(gt = Option(1), ad = Option(Array(35, 10)), gq = Option(7), px = Option(Array(10, 0, 7)))
    val newGenotypes1 = Seq(newGenotype11, newGenotype12, newGenotype13)
    val newRow1 = (newVariant1, (newVa1, newGenotypes1))

    assert(vds.rdd.collect().toSeq == Seq(newRow1))
  }

  @Test def filterFirstOfTwoAllelesSubset(): Unit = {
    val variant1 = new Variant("contig", 0, "ref", IndexedSeq("alt1", "alt2").map(AltAllele("ref", _)))
    val va1 = null
    val genotype11 = Genotype(Option(1), Option(Array(25, 5, 30)), Option(100), Option(5), Option(Array(10, 0, 10, 10, 5, 7)))
    val genotype12 = Genotype(Option(2), Option(Array(25, 35, 0)), Option(100), Option(5), Option(Array(10, 10, 0, 7, 5, 10)))
    val genotype13 = Genotype(Option(3), Option(Array(25, 10, 10)), Option(100), Option(5), Option(Array(10, 10, 10, 0, 5, 7)))
    val genotypes1 = Seq(genotype11, genotype12, genotype13)
    val row1: (Variant, (Annotation, Iterable[Genotype])) = (variant1, (va1, genotypes1))

    val vds = new VariantSampleMatrix(hc, VSMFileMetadata(Array("1", "2", "3"),
      IndexedSeq[Annotation](null, null, null),
      null,
      TString(),
      TStruct.empty),
      sc.parallelize(Seq(row1)).toOrderedRDD)
      .filterAlleles("aIndex == 1", subset = true, keep = false)

    val newVariant1 = variant1.copy(altAlleles = variant1.altAlleles.drop(1))
    val newVa1 = va1
    val newGenotype11 = genotype11.copy(gt = Option(2), ad = Option(Array(25, 30)), gq = Option(3), px = Option(Array(3, 3, 0)))
    val newGenotype12 = genotype12.copy(gt = Option(1), ad = Option(Array(25, 0)), gq = Option(3), px = Option(Array(3, 0, 3)))
    val newGenotype13 = genotype13.copy(gt = Option(1), ad = Option(Array(25, 10)), gq = Option(7), px = Option(Array(10, 0, 7)))
    val newGenotypes1 = Seq(newGenotype11, newGenotype12, newGenotype13)
    val newRow1 = (newVariant1, (newVa1, newGenotypes1))

    assert(vds.rdd.collect().toSeq == Seq(newRow1))
  }

  @Test def filterSecondOfTwoAllelesSubset(): Unit = {
    val variant1 = new Variant("contig", 0, "ref", IndexedSeq("alt1", "alt2").map(AltAllele("ref", _)))
    val va1 = null
    val genotype11 = Genotype(Option(1), Option(Array(25, 5, 30)), Option(100), Option(5), Option(Array(10, 0, 10, 10, 5, 7)))
    val genotype12 = Genotype(Option(2), Option(Array(25, 35, 0)), Option(100), Option(5), Option(Array(10, 10, 0, 7, 5, 10)))
    val genotype13 = Genotype(Option(3), Option(Array(25, 10, 10)), Option(100), Option(5), Option(Array(10, 10, 10, 0, 5, 7)))
    val genotypes1 = Seq(genotype11, genotype12, genotype13)
    val row1: (Variant, (Annotation, Iterable[Genotype])) = (variant1, (va1, genotypes1))

    val vds = new VariantSampleMatrix(hc, VSMFileMetadata(Array("1", "2", "3"),
      IndexedSeq[Annotation](null, null, null),
      null,
      TString(),
      TStruct.empty),
      sc.parallelize(Seq(row1)).toOrderedRDD)
      .filterAlleles("aIndex == 2", subset = true, keep = false)

    val newVariant1 = variant1.copy(altAlleles = variant1.altAlleles.take(1))
    val newVa1 = va1
    val newGenotype11 = genotype11.copy(gt = Option(1), ad = Option(Array(25, 5)), gq = Option(10), px = Option(Array(10, 0, 10)))
    val newGenotype12 = genotype12.copy(gt = Option(2), ad = Option(Array(25, 35)), gq = Option(10), px = Option(Array(10, 10, 0)))
    val newGenotype13 = genotype13.copy(gt = None, ad = Option(Array(25, 10)), gq = Option(0), px = Option(Array(0, 0, 0)))
    val newGenotypes1 = Seq(newGenotype11, newGenotype12, newGenotype13)
    val newRow1 = (newVariant1, (newVa1, newGenotypes1))

    assert(vds.rdd.collect().toSeq == Seq(newRow1))
  }

  @Test def filterSecondOfTwoAllelesFilterAlteredGenotypes(): Unit = {
    val variant1 = new Variant("contig", 0, "ref", IndexedSeq("alt1", "alt2").map(AltAllele("ref", _)))
    val va1 = null
    val genotype11 = Genotype(Option(1), Option(Array(25, 5, 30)), Option(100), Option(5), Option(Array(10, 0, 10, 10, 5, 7)))
    val genotype12 = Genotype(Option(2), Option(Array(25, 35, 0)), Option(100), Option(5), Option(Array(10, 10, 0, 7, 5, 10)))
    val genotype13 = Genotype(Option(3), Option(Array(25, 10, 10)), Option(100), Option(5), Option(Array(10, 10, 10, 0, 5, 7)))
    val genotypes1 = Seq(genotype11, genotype12, genotype13)
    val row1: (Variant, (Annotation, Iterable[Genotype])) = (variant1, (va1, genotypes1))

    val vds = new VariantSampleMatrix(hc, VSMFileMetadata(Array("1", "2", "3"),
      IndexedSeq[Annotation](null, null, null),
      null,
      TString(),
      TStruct.empty),
      sc.parallelize(Seq(row1)).toOrderedRDD)
      .filterAlleles("aIndex == 2", subset = true, keep = false, filterAlteredGenotypes = true)

    val newVariant1 = variant1.copy(altAlleles = variant1.altAlleles.take(1))
    val newVa1 = va1
    val newGenotype11 = genotype11.copy(gt = Option(1), ad = Option(Array(25, 5)), gq = Option(10), px = Option(Array(10, 0, 10)))
    val newGenotype12 = genotype12.copy(gt = Option(2), ad = Option(Array(25, 35)), gq = Option(10), px = Option(Array(10, 10, 0)))
    val newGenotype13 = genotype13.copy(gt = None, ad = Option(Array(25, 10)), gq = Option(0), px = Option(Array(0, 0, 0)))
    val newGenotypes1 = Seq(newGenotype11, newGenotype12, newGenotype13)
    val newRow1 = (newVariant1, (newVa1, newGenotypes1))

    assert(vds.rdd.collect().toSeq == Seq(newRow1))
  }

  @Test def filterOneAlleleAndModifyAnnotation(): Unit = {
    val variant1 = new Variant("contig", 0, "ref", IndexedSeq("alt1", "alt2").map(AltAllele("ref", _)))
    val va1 = null
    val genotypes1 = Seq(Genotype(1), Genotype(2), Genotype(3))
    val row1: (Variant, (Annotation, Iterable[Genotype])) = (variant1, (va1, genotypes1))

    val vds = new VariantSampleMatrix(hc, VSMFileMetadata(Array("1", "2", "3"),
      IndexedSeq[Annotation](null, null, null),
      null,
      TString(),
      TStruct.empty),
      sc.parallelize(Seq(row1)).toOrderedRDD)
      .filterAlleles("aIndex == 2", variantExpr = "va = newToOld", keep = false, subset = false)

    val newVariant1 = variant1.copy(altAlleles = variant1.altAlleles.take(1))
    val newVa1: IndexedSeq[Int] = Array(0, 1)
    val newGenotypes1 = Seq(Genotype(1), Genotype(2), Genotype(0))
    val newRow1 = (newVariant1, (newVa1, newGenotypes1))

    assert(vds.rdd.collect().toSeq == Seq(newRow1))
  }
}
