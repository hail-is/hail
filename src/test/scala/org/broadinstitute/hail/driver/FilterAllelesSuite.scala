package org.broadinstitute.hail.driver

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.check.Prop
import org.broadinstitute.hail.expr.{TChar, TStruct, Type}
import org.broadinstitute.hail.variant.{AltAllele, Genotype, VSMSubgen, Variant, VariantMetadata, VariantSampleMatrix}
import org.scalactic.Equality
import org.testng.annotations.Test
import org.scalatest.Matchers._
import org.broadinstitute.hail.Utils._

class FilterAllelesSuite extends SparkSuite {

  private val noop = "va = va"

  @Test def filterAllAlleles(): Unit = {
    Prop.forAll(VSMSubgen.random.gen(sc)) { vds =>
      val s = FilterAlleles.run(State(sc, sqlContext, vds),
        Array("--keep", "-c", "false", "-a", noop))
      s.vds.nVariants == 0
    }.check()
  }

  @Test def filterNoAlleles(): Unit = {
    Prop.forAll(VSMSubgen.random.gen(sc)) { vds =>
      val s = FilterAlleles.run(State(sc, sqlContext, vds),
        Array("--keep", "-c", "true", "-a", noop))
      s.vds.nVariants == vds.nVariants
    }.check()
  }

  @Test def filterSecondOfTwoAlleles(): Unit = {
    val variant1 = new Variant("contig",0,"ref",IndexedSeq("alt1","alt2").map(AltAllele("ref",_)))
    val va1 = null
    val genotype11 = Genotype(Option(1), Option(Array(25,5,30)), Option(100), Option(5), Option(Array(10,0,10,10,5,7)))
    val genotype12 = Genotype(Option(2), Option(Array(25,35,0)), Option(100), Option(5), Option(Array(10,10,0,7,5,10)))
    val genotype13 = Genotype(Option(3), Option(Array(25,10,10)), Option(100), Option(5), Option(Array(10,10,10,0,5,7)))
    val genotypes1 = Seq(genotype11, genotype12, genotype13)
    val row1: (Variant, (Annotation, Iterable[Genotype])) = (variant1, (va1, genotypes1))

    val vds = VariantSampleMatrix(VariantMetadata(Array("1","2","3"),
      IndexedSeq[Annotation](null,null,null),
      null,
      TChar,
      TStruct.empty,
      TChar),
      sc.parallelize(Seq(row1)).toOrderedRDD)

    val s = FilterAlleles.run(State(sc, sqlContext, vds),
      Array("--remove", "-c", "aIndex == 2", "-a", noop))

    val newVariant1 = variant1.copy(altAlleles = variant1.altAlleles.take(1))
    val newVa1 = va1
    val newGenotype11 = genotype11.copy(ad = Option(Array(55,5)), gq = Option(7), px = Option(Array(7,0,10)))
    val newGenotype12 = genotype12.copy(ad = Option(Array(25,35)), px = Option(Array(7,5,0)))
    val newGenotype13 = genotype13.copy(gt = Option(0), ad = Option(Array(35,10)), px = Option(Array(0,5,10)))
    val newGenotypes1 = Seq(newGenotype11, newGenotype12, newGenotype13)
    val newRow1 = (newVariant1, (newVa1, newGenotypes1))
    s.vds.rdd.collect().toSeq shouldEqual Seq(newRow1)
  }

  @Test def filterFirstOfTwoAlleles(): Unit = {
    val variant1 = new Variant("contig", 0, "ref", IndexedSeq("alt1", "alt2").map(AltAllele("ref", _)))
    val va1 = null
    val genotype11 = Genotype(Option(1), Option(Array(25, 5, 30)), Option(100), Option(5), Option(Array(10, 0, 10, 10, 5, 7)))
    val genotype12 = Genotype(Option(2), Option(Array(25, 35, 0)), Option(100), Option(5), Option(Array(10, 10, 0, 7, 5, 10)))
    val genotype13 = Genotype(Option(3), Option(Array(25, 10, 10)), Option(100), Option(5), Option(Array(10, 10, 10, 0, 5, 7)))
    val genotypes1 = Seq(genotype11, genotype12, genotype13)
    val row1: (Variant, (Annotation, Iterable[Genotype])) = (variant1, (va1, genotypes1))

    val vds = VariantSampleMatrix(VariantMetadata(Array("1", "2", "3"),
      IndexedSeq[Annotation](null, null, null),
      null,
      TChar,
      TStruct.empty,
      TChar),
      sc.parallelize(Seq(row1)).toOrderedRDD)

    val s = FilterAlleles.run(State(sc, sqlContext, vds),
      Array("--remove", "-c", "aIndex == 1", "-a", noop))

    val newVariant1 = variant1.copy(altAlleles = variant1.altAlleles.drop(1))
    val newVa1 = va1
    val newGenotype11 = genotype11.copy(gt = Option(0), ad = Option(Array(30, 30)), px = Option(Array(0, 5, 7)))
    val newGenotype12 = genotype12.copy(gt = Option(0), ad = Option(Array(60, 0)), px = Option(Array(0, 5, 10)))
    val newGenotype13 = genotype13.copy(gt = Option(1), ad = Option(Array(35, 10)), gq = Option(7), px = Option(Array(10, 0, 7)))
    val newGenotypes1 = Seq(newGenotype11, newGenotype12, newGenotype13)
    val newRow1 = (newVariant1, (newVa1, newGenotypes1))
    s.vds.rdd.collect().toSeq shouldEqual Seq(newRow1)
  }

  @Test def filterOneAlleleAndModifyAnnotation(): Unit = {
    val variant1 = new Variant("contig",0,"ref",IndexedSeq("alt1","alt2").map(AltAllele("ref",_)))
    val va1 = null
    val genotypes1 = Seq(Genotype(1), Genotype(2), Genotype(3))
    val row1: (Variant, (Annotation, Iterable[Genotype])) = (variant1, (va1, genotypes1))

    val vds = VariantSampleMatrix(VariantMetadata(Array("1","2","3"),
      IndexedSeq[Annotation](null,null,null),
      null,
      TChar,
      TStruct.empty,
      TChar),
      sc.parallelize(Seq(row1)).toOrderedRDD)

    val s = FilterAlleles.run(State(sc, sqlContext, vds),
      Array("--remove", "-c", "aIndex == 2", "-a", "va = aIndices"))

    val newVariant1 = variant1.copy(altAlleles = variant1.altAlleles.take(1))
    val newVa1: IndexedSeq[Int] = Array(0,1)
    val newGenotypes1 = Seq(Genotype(1), Genotype(2), Genotype(0))
    val newRow1 = (newVariant1, (newVa1, newGenotypes1))

    s.vds.rdd.collect().toSeq shouldEqual Seq(newRow1)
  }

}
