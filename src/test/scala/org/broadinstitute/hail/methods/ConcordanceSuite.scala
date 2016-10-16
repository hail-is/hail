package org.broadinstitute.hail.methods

import org.apache.spark.SparkContext
import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.check.{Gen, Prop}
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.{Genotype, VSMSubgen, Variant, VariantMetadata, VariantSampleMatrix}
import org.testng.annotations.Test

import scala.io.Source
import scala.language._
import scala.sys.process._


class ConcordanceSuite extends SparkSuite {
  def gen(sc: SparkContext) = for (vds1 <- VariantSampleMatrix.gen(sc, VSMSubgen.plinkSafeBiallelic);
    vds2 <- VariantSampleMatrix.gen(sc, VSMSubgen.plinkSafeBiallelic);
    scrambledIds1 <- Gen.shuffle(vds1.sampleIds).map(_.iterator);
    newIds2 <- Gen.parameterized { p =>
      Gen.const(vds2.sampleIds.map { id =>
        if (scrambledIds1.hasNext && p.rng.nextUniform(0, 1) < .5) scrambledIds1.next() else id
      })
    };
    scrambledVariants1 <- Gen.shuffle(vds1.variants.collect()).map(_.iterator);
    newVariantMapping <- Gen.parameterized { p =>
      Gen.const(vds2.variants.collect().map { v =>
        if (scrambledVariants1.hasNext && p.rng.nextUniform(0, 1) < .5) (v, scrambledVariants1.next()) else (v, v)
      }.toMap)
    }
  ) yield (vds1, vds2.copy(sampleIds = newIds2,
    rdd = vds2.rdd.map { case (v, (vaGS)) => (newVariantMapping(v), vaGS) }.toOrderedRDD))

  def readSampleConcordance(file: String): Map[String, IndexedSeq[IndexedSeq[Int]]] = {
    hadoopConf.readLines(file) { lines =>
      lines.filter(line => !line.value.startsWith("sample") && !line.value.startsWith("#Total"))
        .map(_.map { line =>
          val split = line.split("\\s+")
          val sample = split(0)
          val data = (split.tail.init.map(_.toInt): IndexedSeq[Int]).grouped(5).toIndexedSeq
          (sample, data)
        }.value).toMap
    }
  }

  def readVariantConcordance(file: String): Map[Variant, IndexedSeq[IndexedSeq[Int]]] = {
    hadoopConf.readLines(file) { lines =>
      val header = lines.next().value.split("\\s+")

      lines.filter(line => !line.value.startsWith("chr") && !line.value.startsWith("#Total") && !line.value.isEmpty)
        .map(_.map { line =>
          val split = line.split("\\s+")
          val v = Variant(split(0), split(1).toInt, split(2), split(3))
          val data = (split.drop(4).init.map(_.toInt): IndexedSeq[Int]).grouped(5).toIndexedSeq
          (v, data)
        }.value).toMap
    }
  }

  @Test def testSample() {
    var s = ImportVCF.run(State(sc, sqlContext, null), Array("src/test/resources/sample.vcf"))
    s = Filtermulti.run(s)
    s = Put.run(s, Array("-n", "foo")) // Reverse sample IDs
    //    s = Put.run(s.copy(vds = s.vds.copy(sampleIds = s.vds.sampleIds.reverse)), Array("-n", "rev")) // Reverse sample IDs
    //    s = Put.run(s.copy(vds = s.vds.copy(sampleIds = s.vds.sampleIds.reverse)), Array("-n", "foo")) // Reverse sample IDs
    //    s = ExportVCF.run(s.copy(vds = s.vds.copy(sampleIds = s.vds.sampleIds.reverse)), Array("-o", "/Users/tpoterba/data/sample.reversed.vcf")) // Reverse sample IDs
    s = Concordance.run(s, Array("-r", "foo", "-s", "samples", "-v", "variants"))
    s = SampleQC.run(s)

    val (_, nHomRefQ) = s.vds.querySA("sa.qc.nHomRef")
    val (_, nHetQ) = s.vds.querySA("sa.qc.nHet")
    val (_, nHomVarQ) = s.vds.querySA("sa.qc.nHomVar")
    val (_, nNotCalledQ) = s.vds.querySA("sa.qc.nNotCalled")

    case class CallData(nHomRef: Int, nHet: Int, nHomVar: Int, nNotCalled: Int)

    val sampleQcCounts = s.vds.sampleIdsAndAnnotations.toMap.mapValues { a => CallData(
      nHomRefQ(a).get.asInstanceOf[Int],
      nHetQ(a).get.asInstanceOf[Int],
      nHomVarQ(a).get.asInstanceOf[Int],
      nNotCalledQ(a).get.asInstanceOf[Int])
    }
    s = Get.run(s, Array("-n", "samples"))

    val (_, nHomRefQ2) = s.vds.querySA("sa.Left_HomRef_Right_HomRef")
    val (_, nHetQ2) = s.vds.querySA("sa.Left_Het_Right_Het")
    val (_, nHomVarQ2) = s.vds.querySA("sa.Left_HomVar_Right_HomVar")
    val (_, nNotCalledQ2) = s.vds.querySA("sa.Left_Missing_Right_Missing")
    val (_, offDiagSum) = s.vds.querySA(
      """sa.Left_Missing_Right_HomRef +
        |sa.Left_Missing_Right_Het +
        |sa.Left_Missing_Right_HomVar +
        |sa.Left_Missing_Right_NoData +
        |sa.Left_HomRef_Right_Missing +
        |sa.Left_HomRef_Right_Het +
        |sa.Left_HomRef_Right_HomVar +
        |sa.Left_HomRef_Right_NoData +
        |sa.Left_Het_Right_Missing +
        |sa.Left_Het_Right_HomRef +
        |sa.Left_Het_Right_HomVar +
        |sa.Left_Het_Right_NoData +
        |sa.Left_HomVar_Right_Missing +
        |sa.Left_HomVar_Right_HomRef +
        |sa.Left_HomVar_Right_Het +
        |sa.Left_HomVar_Right_NoData +
        |sa.Left_NoData_Right_Missing +
        |sa.Left_NoData_Right_HomRef +
        |sa.Left_NoData_Right_Het +
        |sa.Left_NoData_Right_HomVar +
        |sa.Left_NoData_Right_NoData """.stripMargin)

    val concordCounts = s.vds.sampleIdsAndAnnotations.toMap.mapValues { a =>
      assert(offDiagSum(a).contains(0))
      CallData(
        nHomRefQ2(a).get.asInstanceOf[Int],
        nHetQ2(a).get.asInstanceOf[Int],
        nHomVarQ2(a).get.asInstanceOf[Int],
        nNotCalledQ2(a).get.asInstanceOf[Int])
    }

    assert(concordCounts == sampleQcCounts)

    s = Get.run(s, Array("-n", "variants"))
    PrintSchema.run(s)
  }

  @Test def test() {
    Prop.forAll(gen(sc).filter { case (vds1, vds2) =>
      vds1.sampleIds.toSet.intersect(vds2.sampleIds.toSet).nonEmpty
    }) { case (vds1, vds2) =>
      //      val vds1 = _vds1.filterVariants( { case (v, _, __) => v.start == 1587706484})
      //      val vds2 = _vds1.filterVariants( { case (v, _, __) => v.start == 1587706484})

      //      val vds1 = VariantSampleMatrix(VariantMetadata(Array("sample1")),
      //        sc.parallelize(Seq((Variant("1", 1, "A", "T"), (Annotation.empty, Iterable(Genotype(2)))))).toOrderedRDD).copy(wasSplit = true)
      //
      //      val vds2 = VariantSampleMatrix(VariantMetadata(Array("sample1")),
      //        sc.parallelize(Seq((Variant("1", 1, "A", "T"), (Annotation.empty, Iterable(Genotype(1)))))).toOrderedRDD).copy(wasSplit = true)

      val join = vds1.expand().map { case (v, s, g) => ((v, s), g) }
        .fullOuterJoin(vds2.expand().map { case (v, s, g) => ((v, s), g) })

      val perVariant = join.map { case (k, v) => (k._1, v) }
        .aggregateByKey(new ConcordanceCombiner)({ case (comb, gs) =>
          (gs: @unchecked) match {
            case (Some(g1), Some(g2)) => comb.mergeBoth(g1.unboxedGT, g2.unboxedGT)
            case (Some(g1), None) => comb.mergeLeft(g1.unboxedGT)
            case (None, Some(g2)) => comb.mergeRight(g2.unboxedGT)
          }
          comb
        }, { case (comb1, comb2) => comb1.merge(comb2) })
        .collect()
        .toMap

      val perSample = join.map { case (k, v) => (k._2, v) }
        .aggregateByKey(new ConcordanceCombiner)({ case (comb, gs) =>
          (gs: @unchecked) match {
            case (Some(g1), Some(g2)) => comb.mergeBoth(g1.unboxedGT, g2.unboxedGT)
            case (Some(g1), None) => comb.mergeLeft(g1.unboxedGT)
            case (None, Some(g2)) => comb.mergeRight(g2.unboxedGT)
          }
          comb
        }, { case (comb1, comb2) => comb1.merge(comb2) })
        .collect()
        .toMap
      //      ExportVCF.run(State(sc, sqlContext, vds1), Array("-o", vcf1))
      //      ExportVCF.run(State(sc, sqlContext, vds2), Array("-o", vcf2))
      //
      //      println(s"vds1 has ${ vds1.nVariants } variants")
      //      println(s"vds2 has ${ vds2.nVariants } variants")
      //      val vcf1uri = uriPath(vcf1)
      //      val vcf2uri = uriPath(vcf2)
      //      val variantOut = s"java -Xmx1g -jar /Users/tpoterba/Downloads/snpEff/SnpSift.jar concordance -v $vcf1uri $vcf2uri" !!
      //
      //      val out1 = tmpDir.createLocalTempFile("variantConcordance", "txt")
      //      hadoopConf.writeTextFile(out1)(out => out.write(variantOut))
      //      hadoopConf.readLines(out1)(lines => lines.map(_.value).filter(x => x.contains("chr") || x.contains("1587706484")).foreach(println))
      //
      //      val pwd = System.getProperties.getProperty("user.dir")
      //      val sampleFile = pwd + "/concordance_file1_file2.by_sample.txt"
      //      val variantFile = pwd + "/concordance_file1_file2.by_variant.txt"
      //
      //      val snpSiftSampleConcordance = readSampleConcordance(sampleFile)
      //      println("samples = " + snpSiftSampleConcordance)
      //      val snpSiftVariantConcordance = readVariantConcordance(out1)
      //      println("variants = " + snpSiftVariantConcordance)

      var s = State(sc, sqlContext, vds1, Map("other" -> vds1))
      s = Concordance.run(s, Array("--right", "other", "--samples", "samples", "--variants", "variants"))
      s = Get.run(s, Array("-n", "samples"))
      s.vds.sampleIdsAndAnnotations.foreach { case (s, sa) =>
        println("from concord : " + sa)
        println("from combiner: " + perSample(s).toAnnotation)
        assert(sa == perSample(s).toAnnotation) }

      //      val concordanceQuery1: (Annotation) => IndexedSeq[IndexedSeq[Int]] = {
      //        val (_, query) = s.vds.querySA("sa.concordance")
      //        (sa: Annotation) => query(sa).get.asInstanceOf[IndexedSeq[IndexedSeq[Int]]]
      //      }
      //      s.vds.sampleIdsAndAnnotations.foreach { case (s, sa) =>
      //        assert(snpSiftSampleConcordance(s) == concordanceQuery1(sa))
      //      }

      s = Get.run(s, Array("-n", "variants"))
      s.vds.variantsAndAnnotations.collect().foreach { case (v, va) => assert(va == perVariant(v).toAnnotation) }
      //      val concordanceQuery2: (Annotation) => IndexedSeq[IndexedSeq[Int]] = {
      //        val (_, query) = s.vds.queryVA("va.concordance")
      //        (va: Annotation) => query(va).get.asInstanceOf[IndexedSeq[IndexedSeq[Int]]]
      //      }
      //      s.vds.variantsAndAnnotations.collect().foreach { case (v, va) =>
      //        println(v)
      //        println(s"snpSift=${ snpSiftVariantConcordance(v) }")
      //        println(s"hailCNC=${ concordanceQuery1(va) }")
      //        assert(snpSiftVariantConcordance(v) == concordanceQuery1(va))
      //      }
      //      sys.exit()

      true
    }.check()
  }

}
