package org.broadinstitute.hail.io

import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.check.Gen._
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Properties
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.{FatalException, SparkSuite}
import org.testng.annotations.Test

import scala.io.Source
import scala.language.postfixOps
import scala.sys.process._

class LoadBgenSuite extends SparkSuite {

  def getNumberOfLinesInFile(file: String): Long = {
    readFile(file, sc.hadoopConfiguration) { s =>
      Source.fromInputStream(s)
        .getLines()
        .length
    }.toLong
  }

  def getInfoScoresFromQCTool(statsFile: String): RDD[(Any, Any)] = sc.parallelize(readLines(statsFile, sc.hadoopConfiguration)(_.map(_.transform { line =>
    val Array(snpid, rsid, chromosome, position, a_allele, b_allele, minor_allele,
    major_allele, aa, ab, bb, aa_calls, ab_calls, bb_calls,
    maf, hwe, missing, missing_calls, information) = line.value.split("\\s+")
    (snpid, information)
  }
  ).toArray.drop(1).map { case (s, i) => (s, Option((i.toDouble * 10000).round / 10000.0)) }))

  def getInfoScoresFromSNPTest(statsFile: String): RDD[(Any, Any)] = sc.parallelize(readLines(statsFile, sc.hadoopConfiguration)(_.map(_.transform { line =>
    if (!line.value.startsWith("#")) {
      val x = line.value.split("\\s+")
      (x(0), x(8).toDouble)
    }
  }
  ).toArray.drop(1).map { case (s: String, i: Double) => (s, Option((i * 10000).round / 10000.0)) }))

  @Test def testGavinExample() {
    val gen = "src/test/resources/example.gen"
    val sampleFile = "src/test/resources/example.sample"
    val bgen = "src/test/resources/example.v11.bgen"
    val fileRoot = tmpDir.createTempFile(prefix = "exampleInfoScoreTest")
    val qcToolLogFile = fileRoot + ".qctool.log"
    val statsFile = fileRoot + ".stats"
    val qcToolPath = "qctool"

    hadoopDelete(bgen + ".idx", sc.hadoopConfiguration, true)

    val nSamples = getNumberOfLinesInFile(sampleFile) - 2
    val nVariants = getNumberOfLinesInFile(gen)

    var s = State(sc, sqlContext, null)
    s = IndexBGEN.run(s, Array(bgen))
    s = ImportBGEN.run(s, Array("-s", sampleFile, "-n", "10", bgen))
    assert(s.vds.nSamples == nSamples && s.vds.nVariants == nVariants)

    val genVDS = ImportGEN.run(s, Array("-s", sampleFile, gen)).vds
    val bgenVDS = s.vds

    val varidBgenQuery = bgenVDS.vaSignature.query("varid")
    val rsidBgenQuery = bgenVDS.vaSignature.query("rsid")

    val varidGenQuery = genVDS.vaSignature.query("varid")
    val rsidGenQuery = bgenVDS.vaSignature.query("rsid")

    assert(bgenVDS.metadata == genVDS.metadata)
    assert(bgenVDS.sampleIds == genVDS.sampleIds)

    val bgenAnnotations = bgenVDS.variantsAndAnnotations.map { case (v, va) => (varidBgenQuery(va).get, va) }
    val genAnnotations = genVDS.variantsAndAnnotations.map { case (v, va) => (varidGenQuery(va).get, va) }

    assert(genAnnotations.fullOuterJoin(bgenAnnotations).map { case (varid, (va1, va2)) => if (va1 == va2) true else false }.fold(true)(_ && _))

    val bgenFull = bgenVDS.expandWithAll().map { case (v, va, s, sa, gt) => ((varidBgenQuery(va).get, s), gt) }
    val genFull = genVDS.expandWithAll().map { case (v, va, s, sa, gt) => ((varidGenQuery(va).get, s), gt) }

    genFull.fullOuterJoin(bgenFull)
      .collect()
      .foreach { case ((v, i), (gt1, gt2)) =>
        assert(gt1 == gt2)
      }
  }

  object Spec extends Properties("ImportBGEN") {
    val compGen = for (vds: VariantDataset <- VariantSampleMatrix.gen[Genotype](sc, VSMSubgen.random);
                       nPartitions: Int <- choose(1, 10)) yield (vds, nPartitions)

    val sampleRenameFile = tmpDir.createTempFile(prefix = "sample_rename")
    writeTextFile(sampleRenameFile, sc.hadoopConfiguration) { case w =>
      w.write("NA\tfdsdakfasdkfla")
    }

    property("import generates same output as export") =
      forAll(compGen) { case (vds: VariantSampleMatrix[Genotype], nPartitions: Int) =>

        val vdsRemapped = vds.copy(rdd = vds.rdd.map { case (v, va, gs) => (v.copy(contig = "01"), va, gs) })

        assert(vdsRemapped.rdd.map { case (v, va, gs) =>
          gs.forall{ case g =>
            g.dosage.forall(ad =>
              ad.forall { case d =>
                d >= 0.0 && d <= 1.0
              }
            )
          }
        }.fold(true)(_ && _))

        val fileRoot = tmpDir.createTempFile(prefix = "testImportBgen")
        val sampleFile = fileRoot + ".sample"
        val genFile = fileRoot + ".gen"
        val bgenFile = fileRoot + ".bgen"
        val qcToolLogFile = fileRoot + ".qctool.log"
        val statsFile = fileRoot + ".stats"
        val qcToolPath = "qctool"

        hadoopDelete(bgenFile + ".idx", sc.hadoopConfiguration, true)
        hadoopDelete(bgenFile, sc.hadoopConfiguration, true)
        hadoopDelete(genFile, sc.hadoopConfiguration, true)
        hadoopDelete(sampleFile, sc.hadoopConfiguration, true)
        hadoopDelete(qcToolLogFile, sc.hadoopConfiguration, true)

        var s = State(sc, sqlContext, vdsRemapped)
        s = SplitMulti.run(s, Array[String]())
        s = RenameSamples.run(s, Array("-i", sampleRenameFile))

        val origVds = s.vds

        s = ExportGEN.run(s, Array("-o", fileRoot))

        s"src/test/resources/runExternalToolQuiet.sh $qcToolPath -force -g $genFile -s $sampleFile -og $bgenFile -log $qcToolLogFile" !

        //s"src/test/resources/runExternalToolQuiet.sh $qcToolPath -force -g $genFile -snp-stats $statsFile -log $qcToolLogFile" !

        if (vds.nVariants == 0)
          try {
            s = IndexBGEN.run(s, Array("-n", nPartitions.toString, bgenFile))
            s = ImportBGEN.run(s, Array("-s", sampleFile, "-n", nPartitions.toString, bgenFile))
            false
          } catch {
            case e: FatalException => true
            case _: Throwable => false
          }
        else {
          var q = IndexBGEN.run(State(sc, sqlContext, null), Array(bgenFile))
          q = ImportBGEN.run(State(sc, sqlContext, null), Array("-s", sampleFile, "-n", nPartitions.toString, bgenFile))
          val importedVds = q.vds

          assert(importedVds.nSamples == origVds.nSamples)
          assert(importedVds.nVariants == origVds.nVariants)
          assert(importedVds.sampleIds == origVds.sampleIds)

          val importedVariants = importedVds.variants
          val origVariants = origVds.variants

          val importedFull = importedVds.expandWithAll().map { case (v, va, s, sa, gt) => ((v, s), gt) }
          val originalFull = origVds.expandWithAll().map { case (v, va, s, sa, gt) => ((v, s), gt) }

          originalFull.fullOuterJoin(importedFull).map { case ((v, i), (gt1, gt2)) =>
            if (gt1 == gt2)
              true
            else {
              println(gt1)
              println(gt2)
              if (gt1.isDefined && gt2.isDefined) {
                val gt1x = gt1.get
                val gt2x = gt2.get
                if (gt1x.dosage.zip(gt2x.dosage).map { case (d1, d2) => d1.zip(d2) }
                  .exists(_.forall { case (d1, d2) => math.abs(d1 - d2) <= 3.0e-4 })) {
                  println(s"WARN Not Same Genotype: v=$v i=$i $gt1 $gt2 ${gt1.get.flags} ${gt2.get.flags}")
                  true
                } else {
                  println(s"ERROR Not Same Genotype: v=$v i=$i $gt1 $gt2 ${gt1.get.flags} ${gt2.get.flags}")
                  false
                }
              } else {
                println(s"v=$v i=$i $gt1 $gt2")
                false
              }
            }
          }.fold(true)(_ && _)
        }
      }
  }

  @Test def testBgenImportRandom() {
    Spec.check(100, 100, Option(20))
  }
}