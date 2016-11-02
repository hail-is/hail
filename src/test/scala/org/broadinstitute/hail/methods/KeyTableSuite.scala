package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.{Gen, Properties}
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.keytable.KeyTable
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.{Genotype, VSMSubgen, VariantDataset, VariantSampleMatrix}
import org.testng.annotations.Test

class KeyTableSuite extends SparkSuite {

  @Test def testImportExport() = {
    val inputFile = "src/test/resources/sampleAnnotations.tsv"
    val outputFile = tmpDir.createTempFile("ktImpExp", "tsv")
    var s = State(sc, sqlContext)
    s = ImportKeyTable.run(s, Array("-n", "kt1", "-k", "Sample, Status", inputFile))
    s = ExportKeyTable.run(s, Array("-n", "kt1", "-o", outputFile))

    val importedData = sc.hadoopConfiguration.readLines(inputFile)(_.map(_.value).toIndexedSeq)
    val exportedData = sc.hadoopConfiguration.readLines(outputFile)(_.map(_.value).toIndexedSeq)

    intercept[FatalException] {
      s = ImportKeyTable.run(s, Array("-n", "kt1", "-k", "Sample, Status, BadKeyName", inputFile))
    }

    assert(importedData == exportedData)
  }

  @Test def testFilter() = {
    val data = Array(Array(5, 9, 0), Array(2, 3, 4), Array(1, 2, 3))
    val rdd = sc.parallelize(data.map(Annotation.fromSeq(_)))
    val signature = TStruct(("field1", TInt), ("field2", TInt), ("field3", TInt))
    val keyNames = Array("field1")
    val kt = KeyTable(rdd, signature, keyNames)

    var s = State(sc, sqlContext, ktEnv = Map("kt1" -> kt))

    s = FilterKeyTableExpr.run(s, Array("-n", "kt1", "-c", "field1 < 3", "-d", "kt2", "--keep"))
    assert(s.ktEnv.contains("kt2") && s.ktEnv("kt2").nRows == 2)

    s = FilterKeyTableExpr.run(s, Array("-n", "kt1", "-c", "field1 < 3 && field3 == 4", "-d", "kt3", "--keep"))
    assert(s.ktEnv.contains("kt3") && s.ktEnv("kt3").nRows == 1)

    s = FilterKeyTableExpr.run(s, Array("-n", "kt1", "-c", "field1 == 5 && field2 == 9 && field3 == 0", "-d", "kt3", "--remove"))
    assert(s.ktEnv.contains("kt3") && s.ktEnv("kt3").nRows == 2)

    s = FilterKeyTableExpr.run(s, Array("-n", "kt1", "-c", "field1 < -5 && field3 == 100", "--keep"))
    assert(s.ktEnv.contains("kt1") && s.ktEnv("kt1").nRows == 0)

  }

  def createKey(nItems: Int, nCategories: Int) =
    Gen.buildableOfN[Array, Option[String]](nItems, Gen.option(Gen.oneOfSeq((0 until nCategories).map("group" + _)), 0.95))

  def createKeys(nKeys: Int, nItems: Int) =
    Gen.buildableOfN[Array, Array[Option[String]]](nKeys, createKey(nItems, Gen.choose(1, 10).sample()))

  object Spec extends Properties("CreateKeyTable") {
    val compGen = for (vds: VariantDataset <- VariantSampleMatrix.gen[Genotype](sc, VSMSubgen.random).filter(vds => vds.nVariants > 0 && vds.nSamples > 0);
      nKeys <- Gen.choose(1, 5);
      nSampleKeys <- Gen.choose(0, nKeys);
      nVariantKeys <- Gen.const(nKeys - nSampleKeys);
      sampleGroups <- createKeys(nSampleKeys, vds.nSamples);
      variantGroups <- createKeys(nVariantKeys, vds.nVariants.toInt)
    ) yield (vds, sampleGroups, variantGroups)

    property("aggregate by sample and variants same") = forAll(compGen) { case (vds, sampleGroups, variantGroups) =>
      val outputFile = tmpDir.createTempFile("keyTableTest", "tsv")

      val nKeys = sampleGroups.length + variantGroups.length
      val keyNames = (1 to nKeys).map("key" + _)
      val sampleKeyNames = (1 to sampleGroups.length).map("key" + _)
      val variantKeyNames = (sampleGroups.length + 1 to nKeys).map("key" + _)

      var sampleSignature = TStruct()
      sampleKeyNames.foreach(k => sampleSignature = sampleSignature.appendKey(k, TString))

      var variantSignature = TStruct()
      variantKeyNames.foreach(k => variantSignature = variantSignature.appendKey(k, TString))

      val sampleMap = vds.sampleIds.zipWithIndex.map { case (sid, i) =>
        (sid, Annotation(sampleGroups.map(_ (i).orNull).toSeq: _*))
      }.toMap

      val variantAnnotations = sc.parallelize(vds.variants.collect().zipWithIndex.map { case (v, i) =>
        (v, Annotation(variantGroups.map(_ (i).orNull).toSeq: _*))
      }).toOrderedRDD

      var s = State(sc, sqlContext, vds.annotateSamples(sampleMap, sampleSignature, "sa.keys")
        .annotateVariants(variantAnnotations, variantSignature, "va.keys"))

      val (_, sampleKeyQuery) = s.vds.querySA("sa.keys.*")
      val (_, variantKeyQuery) = s.vds.queryVA("va.keys.*")

      val keyGenotypeRDD = s.vds.mapWithAll { case (v, va, sid, sa, g) =>
        val key = sampleKeyQuery(sa).get.asInstanceOf[IndexedSeq[String]] ++ variantKeyQuery(va).get.asInstanceOf[IndexedSeq[String]]
        (key, g)
      }

      val result = keyGenotypeRDD.aggregateByKey((0L, 0L, 0L))(
        (comb, gt) => (comb._1 + gt.isHet.toInt.toInt, comb._2 + gt.isCalled.toInt.toInt, comb._3 + 1),
        (comb1, comb2) => (comb1._1 + comb2._1, comb1._2 + comb2._2, comb1._3 + comb2._3)).collectAsMap()

      s = AggregateByKey.run(s, Array("-k", (sampleKeyNames.map(k => k + " = " + "sa.keys." + k) ++ variantKeyNames.map(k => k + " = " + "va.keys." + k)).mkString(","),
        "-a", "nHet = gs.filter(g => g.isHet).count(), nCalled = gs.filter(g => g.isCalled).count(), nTotal = gs.count()", "-n", "foo"))

      s = ExportKeyTable.run(s, Array("-o", outputFile, "-n", "foo"))
      println(outputFile)

      val ktr = hadoopConf.readLines(outputFile)(_.map(_.map { line =>
        line.trim.split("\\s+")
      }.value).toIndexedSeq)

      val header = ktr.take(1)

      val keyTableResults = ktr.drop(1).map(r => (header(0), r).zipped.toMap)
        .map { x => (keyNames.map { k => x(k) }, (x("nHet"), x("nCalled"), x("nTotal"))) }.toMap

      result.forall { case (keys, (nHet, nCalled, nTotal)) =>
        val (ktHet, ktCalled, ktTotal) = keyTableResults(keys.map(k => if (k != null) k else "NA").toIndexedSeq)
        ktHet.toLong == nHet &&
          ktCalled.toLong == nCalled &&
          ktTotal.toLong == nTotal
      }
    }
  }
//
//  @Test def testAddKeyTable() {
//    Spec.check()
//  }

}
