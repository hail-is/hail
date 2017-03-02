package is.hail.methods

import is.hail.SparkSuite
import is.hail.check._
import is.hail.expr._
import is.hail.variant.{GTPair, GenericGenotype, Genotype, VSMSubgen, Variant, VariantDataset, VariantSampleMatrix}
import org.apache.commons.math3.distribution.{BinomialDistribution, NormalDistribution}
import org.testng.annotations.Test

class DeNovoSuite extends SparkSuite {

  @Test def test() {

    val kt = hc.read("/Users/tpoterba/data/denovo/TT_esp_ready.vds")
      .filterVariantsExpr("va.pass")
      .filterVariantsExpr("v.start < 825280 ")
      .filterMulti()
      .deNovo("/Users/tpoterba/data/denovo/TT.fam", "va.esp")

    kt.rdd.collect().foreach(println)
    println(kt.nRows)
  }

  val gen: Gen[(VariantDataset, Pedigree)] = {
    for {
      vds <- VariantSampleMatrix.gen[Genotype](hc, VSMSubgen.plinkSafeBiallelic.copy(
        saSigGen = Gen.const(TStruct.empty),
        vaSigGen = Gen.const(TStruct.empty),
        globalSigGen = Gen.const(TStruct.empty),
        vGen = for {
          pos <- Gen.choose(1, 1000)
          alt <- Gen.oneOf("T", "AA") // represent indels and snps
          contig <- Gen.oneOf("1", "X", "Y")
        } yield Variant(contig, pos, "A", alt), tGen = (i: Int) => {
          require(i == 2)
          val alleleFrequency = Gen.choose(0.01, 0.30).sample()
          val readBias = Gen.choose(0.01, 0.10).sample()
          val gqMultiplier = Gen.choose(1d, 5d).sample()
          val avgDepth = Gen.choose(20d, 40d).sample()
          val depthSD = avgDepth / 2

          val depthGenerator = new NormalDistribution(avgDepth, depthSD)

          def sampleDepth(): Int = math.max(depthGenerator.sample(), 0).round.toInt

          def sampleAD(dp: Int, pAlt: Double): Array[Int] = {
            val altReads = new BinomialDistribution(dp, pAlt).sample()
            Array(dp - altReads, altReads)
          }


          def allele(d: Double): Int = if (d < alleleFrequency) 1 else 0

          for {
            isMissing <- Gen.choose(0d, 1d).map(_ < 0.1)
            a1 <- Gen.choose(0d, 1d).map(allele)
            a2 <- Gen.choose(0d, 1d).map(allele)
            gt <- Gen.const(a1 + a2)
            dp <- Gen.const(sampleDepth())
            ad <- Gen.const(if (gt == 0)
              sampleAD(dp, readBias)
            else if (gt == 1)
              sampleAD(dp, 0.5)
            else
              sampleAD(dp, 1 - readBias))
            gq <- Gen.const(if (gt == 0 || gt == 1) (gqMultiplier * dp).round.toInt
            else (dp * gqMultiplier * 3).round.toInt)
            pls <- Gen.const(if (gt == 0)
              Array(0, gq, gq * 3)
            else if (gt == 1)
              Array(gq, 0, gq)
            else
              Array(gq * 3, gq, 0))
          } yield if (isMissing) Genotype() else new GenericGenotype(gt, ad, dp, gq, pls, false, false)
        },
        nSampleOverride = Some(100),
        nVariantOverride = Some(100)
      ))
      ped <- Pedigree.gen(vds.sampleIds, completeTrios = true)
    } yield (vds, ped)
  }

  @Test def test2() {
    val (vds, ped) = gen.resize(100000).sample()

    val vds2 = vds.annotateVariantsExpr("va.callStats = gs.callStats(g => v)")
      .annotateVariantsExpr("va.esp = max(va.callStats.AF[1] + runif(-0.05, 0.05), 0)")
      .cache()
    vds2.exportVariants("/tmp/fake_esp.txt",
      "Chromosome = v.contig, Position = v.start, Ref = v.ref, Alt = v.alt, 4 = 0, 5 = 0, " +
        "6 = 0, 7 = 0, 8 = 0, AC_EA = va, AN_EA = 1, 11 = 0, AC_AA = 0, AN_AA = 0, 12 = 0")

    println(vds2.queryVariants("variants.map(v => {v: v, va:va}).collect()"))
    vds2.exportVCF("/tmp/test_realistic.vcf")
    ped.write("/tmp/test.fam", hc.hadoopConf)

    ped.writeSummary("/tmp/summary", hc.hadoopConf)
    val kt = vds2.deNovo("/tmp/test.fam", "va.esp")

    println(s"found ${kt.nRows} de novo events")
    kt.rdd.collect().foreach(println)
  }
}
