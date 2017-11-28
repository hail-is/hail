package is.hail.methods

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.check._
import is.hail.expr._
import is.hail.utils._
import is.hail.variant.{GenericGenotype, Genotype, Locus, VSMSubgen, Variant, VariantSampleMatrix}
import org.apache.commons.math3.distribution.{BinomialDistribution, NormalDistribution}
import org.testng.annotations.Test

import scala.sys.process._

class DeNovoSuite extends SparkSuite {
  /**
    * We version-control the entire de novo script in the resources directory. In order to test robustly
    * against the functionality of the de_novo_finder.py script, we randomly generate realistic-looking
    * data, then generate a trio, and compute de novo calls using Hail's implementation. We also export
    * this dataset as a VCF and run the de novo caller on this output, analyzing the results to determine
    * that the results agree.
    *
    * Since the random tests are slow, we disable them by default. We check in one generated VCF and .fam
    * combination that produces a good distribution of de novo calls, and test against only this VCF.
    */
  lazy val gen: Gen[(VariantSampleMatrix, Pedigree)] = {
    for {
      vds <- VariantSampleMatrix.gen(hc, VSMSubgen.plinkSafeBiallelic.copy(
        saSigGen = Gen.const(TStruct.empty()),
        vaSigGen = Gen.const(TStruct.empty()),
        globalSigGen = Gen.const(TStruct.empty()),
        vGen = (TVariant) => (for {
          pos <- Gen.choose(100000, 100000000)
          alt <- Gen.oneOf("T", "AA") // represent indels and snps
          contig <- Gen.oneOf("1", "X", "Y") // no MT (python caller doesn't support that)
        } yield Variant(contig, pos, "A", alt)).filter(v => !v.inYPar),
        tGen = (TGenotype, v: Annotation) => {
          val alleleFrequency = Gen.choose(0.001, 0.10).sample()
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
          } yield if (isMissing) Genotype() else new GenericGenotype(gt, ad, dp, math.min(gq, 99), pls)
        }
      )).filter(vds => vds.countVariants() > 0 && vds.nSamples > 0)
      ped <- Pedigree.gen(vds.stringSampleIds)
    } yield (vds, ped)
  }

  def check(pathBase: String) {
    val vcfURI = uriPath(pathBase + ".vcf")
    val famURI = uriPath(pathBase + ".fam")
    val espURI = uriPath(pathBase + ".esp")


    val callerOutputString = s"python src/test/resources/de_novo_finder_3.py $vcfURI $famURI $espURI" !!

    val cm = callerOutputString.split("\n")
      .iterator
      .dropWhile(l => l.startsWith("#") || l.startsWith("Chr"))
      .map { l =>
        val line = l.split("\t")
        val chr = line(0)
        val pos = line(1).toInt
        val ref = line(3)
        val alt = line(4)
        val kid = line(5)
        val pdn = line(20).toDouble
        val anno = line(21)
        ((Variant(chr, pos, ref, alt), kid), (pdn, anno))
      }.toMap

    val vds = hc.importVCF(pathBase + ".vcf")
      .verifyBiallelic()
    val kt = vds.deNovo(Pedigree.read(pathBase + ".fam", hadoopConf), "va.info.ESP")

    kt.typeCheck()

    val ktOut = kt.rdd
      .map { r =>
        val v = r.getAs[Variant](0)
        val kid = r.getAs[String](1)
        val pdn = r.getAs[Double](9)
        val isFemale = r.getAs[Boolean](4)
        val anno = r.getAs[String](5)
        ((v, kid), (pdn, anno, isFemale))
      }.collect()

    val p1 = ktOut.forall { case ((v, kid), (pdn, anno, isf)) =>
      val variantRoot = if (v.altAllele.isSNP) "_SNV" else "_indel"
      val adjAnno = anno + variantRoot
      cm.get((v, kid)) match {
        case Some((cPdn, cAnno)) =>
        val p = D_==(cPdn, pdn, tolerance = 1e-4) && cAnno == adjAnno
        if (!p)
          println(s"MISMATCH: ${ v }, $kid, ($cPdn/$pdn), ($cAnno/$adjAnno), isFemale=$isf")
        p
        case None => println(s"NO DeNovoFinder MATCH FOUND: ${v -> kid}, ${pdn -> anno}")
          false
      }
    }
    val p2 = cm.keySet == ktOut.map(_._1).toSet
    println(s"caller out: ${ cm.size }, ${ cm }")
    println(s"dn out: ${ ktOut.size }, ${ ktOut.toSeq }")
    println(s"set diffs: ${ cm.keySet.diff(ktOut.map(_._1).toSet) }")
    assert(p1 && p2)
  }

  def writeAll(vds: VariantSampleMatrix, ped: Pedigree, path: String) {
    val annot = vds.annotateVariantsExpr("va.callStats = gs.callStats(g => v), va.filters = [\"PASS\"].toSet")
      .filterVariantsExpr("va.callStats.AC[1] > 0")
      .annotateVariantsExpr(
        "va.info.AC = va.callStats.AC[1:], va.info.AN=va.callStats.AN, va.info.AF = va.callStats.AF[1:]," +
          "va.info.ESP = min(max(va.callStats.AF[1] + runif(-0.05, 0.05), 0), 1)")
      .cache()
    annot.exportVCF(path + ".vcf")
    annot.variantsKT().select("Chromosome = v.contig", "Position = v.start", "Ref = v.ref", "Alt = v.alt", "4 = 0", "5 = 0",
    "6 = 0", "7 = 0", "8 = 0", "AC_EA = va.info.ESP", "AN_EA = 1", "11 = 0", "AC_AA = 0", "AN_AA = 0", "12 = 0")
        .export(path + ".esp")
    ped.write(path + ".fam", hc.hadoopConf)
  }

  @Test def test() {
    check("src/test/resources/denovo")
  }

  @Test(enabled = false) def testRandom() {
    Prop.forAll(gen.resize(1000000)) { case (vds, ped) =>
      val tmpFile = tmpDir.createTempFile()

      writeAll(vds, ped, tmpFile)

      check(tmpFile)

      // 'check' raises assertion errors for failures
      true
    }.check()
  }
}
