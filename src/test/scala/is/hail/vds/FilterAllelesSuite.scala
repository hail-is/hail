package is.hail.vds

import is.hail.SparkSuite
import is.hail.annotations._
import is.hail.check.Prop
import is.hail.expr.types._
import is.hail.testUtils._
import is.hail.variant.{AltAllele, Genotype, MatrixFileMetadata, MatrixTable, VSMSubgen, Variant}
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class FilterAllelesSuite extends SparkSuite {
  @Test def filterSecondOfTwoAllelesDowncode(): Unit = {
    val variant1 = new Variant("contig", 0, "ref", IndexedSeq("alt1", "alt2").map(AltAllele("ref", _)))
    val va1 = Row(variant1)
    val genotype11 = Genotype(Option(1), Option(Array(25, 5, 30)), Option(100), Option(5), Option(Array(10, 0, 10, 10, 5, 7)))
    val genotype12 = Genotype(Option(2), Option(Array(25, 35, 0)), Option(100), Option(5), Option(Array(10, 10, 0, 7, 5, 10)))
    val genotype13 = Genotype(Option(3), Option(Array(25, 10, 10)), Option(100), Option(5), Option(Array(10, 10, 10, 0, 5, 7)))
    val genotypes1 = Seq(genotype11, genotype12, genotype13)
    val row1: (Annotation, Iterable[Annotation]) = (va1, genotypes1)

    val vds = MatrixTable.fromLegacy(hc, MatrixFileMetadata(sampleAnnotations = Array("1", "2", "3").map(Annotation(_))),
      sc.parallelize(Seq(row1)))
      .filterAlleles("aIndex == 2", subset = false, keep = false)

    val newVariant1 = variant1.copy(altAlleles = variant1.altAlleles.take(1))
    val newVa1 = Row(newVariant1.locus, newVariant1.alleles)
    val newGenotype11 = Genotype(Option(1), Option(Array(55, 5)), Option(100), Option(7), Option(Array(7, 0, 10)))
    val newGenotype12 = Genotype(Option(2), Option(Array(25, 35)), Option(100), Option(5), Option(Array(7, 5, 0)))
    val newGenotype13 = Genotype(Option(0), Option(Array(35, 10)), Option(100), Option(5), Option(Array(0, 5, 10)))
    val newGenotypes1 = Seq(newGenotype11, newGenotype12, newGenotype13)
    val newRow1 = (newVariant1, (newVa1, newGenotypes1))
    println(Seq(newRow1).mkString(","))
    println(vds.variantRDD.collect().toSeq.mkString(","))

    assert(vds.variantRDD.collect().toSeq == Seq(newRow1))
  }

  @Test def filterFirstOfTwoAllelesDowncode(): Unit = {
    val variant1 = new Variant("contig", 0, "ref", IndexedSeq("alt1", "alt2").map(AltAllele("ref", _)))
    val va1 = Row(variant1)
    val genotype11 = Genotype(Option(1), Option(Array(25, 5, 30)), Option(100), Option(5), Option(Array(10, 0, 10, 10, 5, 7)))
    val genotype12 = Genotype(Option(2), Option(Array(25, 35, 0)), Option(100), Option(5), Option(Array(10, 10, 0, 7, 5, 10)))
    val genotype13 = Genotype(Option(3), Option(Array(25, 10, 10)), Option(100), Option(5), Option(Array(10, 10, 10, 0, 5, 7)))
    val genotypes1 = Seq(genotype11, genotype12, genotype13)
    val row1: (Annotation, Iterable[Annotation]) = (va1, genotypes1)

    val vds = MatrixTable.fromLegacy(hc, MatrixFileMetadata(Array("1", "2", "3").map(Annotation(_))),
      sc.parallelize(Seq(row1)))
      .filterAlleles("aIndex == 1", subset = false, keep = false)

    val newVariant1 = variant1.copy(altAlleles = variant1.altAlleles.drop(1))
    val newVa1 = Row(newVariant1.locus, newVariant1.alleles)

    val newGenotype11 = Genotype(Option(0), Option(Array(30, 30)), Option(100), Option(5), Option(Array(0, 5, 7)))
    val newGenotype12 = Genotype(Option(0), Option(Array(60, 0)), Option(100), Option(5), Option(Array(0, 5, 10)))
    val newGenotype13 = Genotype(Option(1), Option(Array(35, 10)), Option(100), Option(7), Option(Array(10, 0, 7)))
    val newGenotypes1 = Seq(newGenotype11, newGenotype12, newGenotype13)
    val newRow1 = (newVariant1, (newVa1, newGenotypes1))

    println(Seq(newRow1).mkString(","))
    println(vds.variantRDD.collect().toSeq.mkString(","))
    assert(vds.variantRDD.collect().toSeq == Seq(newRow1))
  }

  @Test def filterFirstOfTwoAllelesSubset(): Unit = {
    val variant1 = new Variant("contig", 0, "ref", IndexedSeq("alt1", "alt2").map(AltAllele("ref", _)))
    val va1 = Row(variant1)
    val genotype11 = Genotype(Option(1), Option(Array(25, 5, 30)), Option(100), Option(5), Option(Array(10, 0, 10, 10, 5, 7)))
    val genotype12 = Genotype(Option(2), Option(Array(25, 35, 0)), Option(100), Option(5), Option(Array(10, 10, 0, 7, 5, 10)))
    val genotype13 = Genotype(Option(3), Option(Array(25, 10, 10)), Option(100), Option(5), Option(Array(10, 10, 10, 0, 5, 7)))
    val genotypes1 = Seq(genotype11, genotype12, genotype13)
    val row1: (Annotation, Iterable[Annotation]) = (va1, genotypes1)

    val vds = MatrixTable.fromLegacy(hc, MatrixFileMetadata(Array("1", "2", "3").map(Annotation(_))),
      sc.parallelize(Seq(row1)))
      .filterAlleles("aIndex == 1", subset = true, keep = false)

    val newVariant1 = variant1.copy(altAlleles = variant1.altAlleles.drop(1))
    val newVa1 = Row(newVariant1.locus, newVariant1.alleles)
    val newGenotype11 = Genotype(Option(2), Option(Array(25, 30)), Option(100), Option(3), Option(Array(3, 3, 0)))
    val newGenotype12 = Genotype(Option(1), Option(Array(25, 0)), Option(100), Option(3), Option(Array(3, 0, 3)))
    val newGenotype13 = Genotype(Option(1), Option(Array(25, 10)), Option(100), Option(7), Option(Array(10, 0, 7)))
    val newGenotypes1 = Seq(newGenotype11, newGenotype12, newGenotype13)
    val newRow1 = (newVariant1, (newVa1, newGenotypes1))

    assert(vds.variantRDD.collect().toSeq == Seq(newRow1))
  }

  @Test def filterSecondOfTwoAllelesSubset(): Unit = {
    val variant1 = new Variant("contig", 0, "ref", IndexedSeq("alt1", "alt2").map(AltAllele("ref", _)))
    val va1 = Row(variant1)
    val genotype11 = Genotype(Option(1), Option(Array(25, 5, 30)), Option(100), Option(5), Option(Array(10, 0, 10, 10, 5, 7)))
    val genotype12 = Genotype(Option(2), Option(Array(25, 35, 0)), Option(100), Option(5), Option(Array(10, 10, 0, 7, 5, 10)))
    val genotype13 = Genotype(Option(3), Option(Array(25, 10, 10)), Option(100), Option(5), Option(Array(10, 10, 10, 0, 5, 7)))
    val genotypes1 = Seq(genotype11, genotype12, genotype13)
    val row1: (Annotation, Iterable[Annotation]) = (va1, genotypes1)

    val vds = MatrixTable.fromLegacy(hc, MatrixFileMetadata(Array("1", "2", "3").map(Annotation(_))),
      sc.parallelize(Seq(row1)))
      .filterAlleles("aIndex == 2", subset = true, keep = false)

    val newVariant1 = variant1.copy(altAlleles = variant1.altAlleles.take(1))
    val newVa1 = Row(newVariant1.locus, newVariant1.alleles)
    val newGenotype11 = Genotype(Option(1), Option(Array(25, 5)), Option(100), Option(10), Option(Array(10, 0, 10)))
    val newGenotype12 = Genotype(Option(2), Option(Array(25, 35)), Option(100), Option(10), Option(Array(10, 10, 0)))
    val newGenotype13 = Genotype(None, Option(Array(25, 10)), Option(100), Option(0), Option(Array(0, 0, 0)))
    val newGenotypes1 = Seq(newGenotype11, newGenotype12, newGenotype13)
    val newRow1 = (newVariant1, (newVa1, newGenotypes1))

    assert(vds.variantRDD.collect().toSeq == Seq(newRow1))
  }
  
  @Test def filterOneAlleleAndModifyAnnotation(): Unit = {
    val variant1 = new Variant("contig", 0, "ref", IndexedSeq("alt1", "alt2").map(AltAllele("ref", _)))
    val va1 = Row(variant1)
    val genotypes1 = Seq(Genotype(1), Genotype(2), Genotype(3))
    val row1: (Annotation, Iterable[Annotation]) = (va1, genotypes1)

    val vds = MatrixTable.fromLegacy(hc, MatrixFileMetadata(Array("1", "2", "3").map(Annotation(_))),
      sc.parallelize(Seq(row1)))
      .filterAlleles("aIndex == 2", variantExpr = "va = {locus: newLocus, alleles: newAlleles, nto: newToOld}", keep = false, subset = false)

    val newVariant1 = variant1.copy(altAlleles = variant1.altAlleles.take(1))
    val newVa1 = Row(newVariant1.locus, newVariant1.alleles, IndexedSeq(0, 1))
    val newGenotypes1 = Seq(Genotype(1), Genotype(2), Genotype(0))
    val newRow1 = (newVariant1, (newVa1, newGenotypes1))

    assert(vds.variantRDD.collect().toSeq == Seq(newRow1))
  }
}
