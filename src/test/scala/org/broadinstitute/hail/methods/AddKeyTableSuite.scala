package org.broadinstitute.hail.methods

import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test
import org.broadinstitute.hail.driver._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.check.Arbitrary._
import org.broadinstitute.hail.variant.{Genotype, VSMSubgen, VariantDataset, VariantSampleMatrix}
import org.broadinstitute.hail.check.Gen
import org.broadinstitute.hail.check.Prop._
import org.broadinstitute.hail.check.Properties
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.utils.TextTableReader

class AddKeyTableSuite extends SparkSuite {

  def createKey(nItems: Int, nCategories: Int) =
    Gen.buildableOfN[Array, Option[String]](nItems, Gen.option(Gen.oneOfSeq((0 until nCategories).map("group" + _)), 0.95))

  def createKeys(nKeys: Int, nItems: Int) =
    Gen.buildableOfN[Array, Array[Option[String]]](nKeys, createKey(nItems, Gen.choose(1, 10).sample()))

//  @Test def test1() {
//    var s = State(sc, sqlContext, null)
//    s = ImportVCF.run(s, Array("-i", "src/test/resources/sample.vcf"))
//    s = AnnotateVariantsExpr.run(s, Array("-c", "va.foo = gs.filter(g => g.isHet).count()"))
//    s = AnnotateSamplesExpr.run(s, Array("-c", "sa.foo = gs.filter(g => g.isHet).count()"))
//    s = AnnotateGlobalExpr.run(s, Array("-c", "global.foo = variants.count()"))
//    s = AddKeyTable.run(s, Array("-k", "foo1 = va.foo, foo2 = sa.foo", "-a", "hetCount = gs.filter(g => g.isHet).count(), totalCount = gs.count()", "-o", "testKeyTable.tsv"))
// }

  object Spec extends Properties("CreateKeyTable") {
    val compGen = for (vds: VariantDataset <- VariantSampleMatrix.gen[Genotype](sc, VSMSubgen.random).filter(vds => vds.nVariants > 0 && vds.nSamples > 0);
      nKeys <- Gen.choose(1, 5);
      nSampleKeys <- Gen.choose(0, nKeys);
      nVariantKeys <- Gen.const(nKeys - nSampleKeys);
      sampleGroups <- createKeys(nSampleKeys, vds.nSamples);
      variantGroups <- createKeys(nVariantKeys, vds.nVariants.toInt)
    ) yield (vds, sampleGroups, variantGroups)

    val compGenSample = for (vds: VariantDataset <- VariantSampleMatrix.gen[Genotype](sc, VSMSubgen.random).filter(vds => vds.nVariants > 0 && vds.nSamples > 0);
      nSampleKeys <- Gen.choose(2, 5);
      sampleGroups <- createKeys(nSampleKeys, vds.nSamples)
    ) yield (vds, sampleGroups)

    val compGenVariant = for (vds: VariantDataset <- VariantSampleMatrix.gen[Genotype](sc, VSMSubgen.random).filter(vds => vds.nVariants > 0 && vds.nSamples > 0);
      nVariantKeys <- Gen.choose(2, 5);
      variantGroups <- createKeys(nVariantKeys, vds.nVariants.toInt)
    ) yield (vds, variantGroups)

    def getKeyTableResults(fileName: String, keyNames: IndexedSeq[String]) = {
      val ktr = hadoopConf.readLines(fileName)(_.map(_.map { line =>
        line.trim.split("\\s+")
      }.value).toIndexedSeq)

      val header = ktr.take(1)
      ktr.drop(1).map(r => (header(0), r).zipped.toMap)
        .map { x => (keyNames.map { k => x(k) }, (x("nHet"), x("nCalled"), x("nTotal"))) }.toMap
    }

    def keyTableEqualAnnExpr(annExprResult: scala.collection.Map[IndexedSeq[String], (Long, Long, Long)], keyTableResults: scala.collection.Map[IndexedSeq[String], (String, String, String)]) =
      annExprResult.forall{ case (keys, (nHet, nCalled, nTotal)) =>
        val (ktHet, ktCalled, ktTotal) = keyTableResults(keys.map(k => if (k != null) k else "NA").toIndexedSeq)
        ktHet.toLong == nHet &&
          ktCalled.toLong == nCalled &&
          ktTotal.toLong == nTotal
      }

    property("group by variant id same as variant aggregations") = forAll(VariantSampleMatrix.gen[Genotype](sc, VSMSubgen.random).filter(vds => vds.nVariants > 0 && vds.nSamples > 0)) {case (vds: VariantDataset) =>
      val outputFile = tmpDir.createTempFile("aggByVariant", ".tsv")

      var s = State(sc, sqlContext, vds)

      s = AnnotateVariantsExpr.run(s, Array("-c", "va.nHet = gs.filter(g => g.isHet).count(), va.nCalled = gs.filter(g => g.isCalled).count(), va.nTotal = gs.count()"))

      val (_, nHetQuery) = s.vds.queryVA("va.nHet")
      val (_, nCalledQuery) = s.vds.queryVA("va.nCalled")
      val (_, nTotalQuery) = s.vds.queryVA("va.nTotal")

      val truthResult = s.vds.variantsAndAnnotations.map{ case (v, va) =>
        (IndexedSeq(v.toString), (nHetQuery(va).get.asInstanceOf[Long], nCalledQuery(va).get.asInstanceOf[Long], nTotalQuery(va).get.asInstanceOf[Long]))
      }.collectAsMap()

      s = AddKeyTable.run(s, Array("-k", "Variant = v",
        "-a", "nHet = gs.filter(g => g.isHet).count(), nCalled = gs.filter(g => g.isCalled).count(), nTotal = gs.count()",
        "-o", outputFile))

      val keyTableResults = getKeyTableResults(outputFile, IndexedSeq("Variant"))

      keyTableEqualAnnExpr(truthResult, keyTableResults)
    }

    property("group by sample id same as sample aggregations") = forAll(VariantSampleMatrix.gen[Genotype](sc, VSMSubgen.random).filter(vds => vds.nVariants > 0 && vds.nSamples > 0)) {case (vds: VariantDataset) =>
      val outputFile = tmpDir.createTempFile("aggBySample", ".tsv")

      var s = State(sc, sqlContext, vds)

      s = AnnotateSamplesExpr.run(s, Array("-c", "sa.nHet = gs.filter(g => g.isHet).count(), sa.nCalled = gs.filter(g => g.isCalled).count(), sa.nTotal = gs.count()"))

      val (_, nHetQuery) = s.vds.querySA("sa.nHet")
      val (_, nCalledQuery) = s.vds.querySA("sa.nCalled")
      val (_, nTotalQuery) = s.vds.querySA("sa.nTotal")

      val truthResult = s.vds.sampleIdsAndAnnotations.map{ case (sid, sa) =>
        (IndexedSeq(sid), (nHetQuery(sa).get.asInstanceOf[Long], nCalledQuery(sa).get.asInstanceOf[Long], nTotalQuery(sa).get.asInstanceOf[Long]))
      }.toMap

      s = AddKeyTable.run(s, Array("-k", "Sample = s",
        "-a", "nHet = gs.filter(g => g.isHet).count(), nCalled = gs.filter(g => g.isCalled).count(), nTotal = gs.count()",
        "-o", outputFile))

      val keyTableResults = getKeyTableResults(outputFile, IndexedSeq("Sample"))

      keyTableEqualAnnExpr(truthResult, keyTableResults)
    }

    property("aggregate by variant groups same") = forAll(compGenVariant) { case (vds, varGroups) =>
      val outputFile = tmpDir.createTempFile("aggByVariantGroup", ".tsv")

      val nKeys = varGroups.length
      val keyNames = (1 to nKeys).map("key" + _)

      var signature = TStruct()
      keyNames.foreach(k => signature = signature.appendKey(k, TString))

      val variantAnnotations = sc.parallelize(vds.variants.collect().zipWithIndex.map { case (v, i) =>
        (v, Annotation(varGroups.map(_ (i).orNull).toSeq: _*))
      }).toOrderedRDD

      var s = State(sc, sqlContext, vds.annotateVariants(variantAnnotations, signature, "va.keys"))

      s = AnnotateVariantsExpr.run(s, Array("-c", "va.nHet = gs.filter(g => g.isHet).count(), va.nCalled = gs.filter(g => g.isCalled).count(), va.nTotal = gs.count()"))

      val (_, nHetQuery) = s.vds.queryVA("va.nHet")
      val (_, nCalledQuery) = s.vds.queryVA("va.nCalled")
      val (_, nTotalQuery) = s.vds.queryVA("va.nTotal")
      val (_, keyQuery) = s.vds.queryVA("va.keys.*")

      val truthResult = s.vds.variantsAndAnnotations.map{ case (v, va) =>
        (keyQuery(va).get.asInstanceOf[IndexedSeq[String]], (nHetQuery(va).get.asInstanceOf[Long], nCalledQuery(va).get.asInstanceOf[Long], nTotalQuery(va).get.asInstanceOf[Long]))
      }.aggregateByKey((0L, 0L, 0L))((comb, counts) => (comb._1 + counts._1, comb._2 + counts._2, comb._3 + counts._3),
        (comb1, comb2) => (comb1._1 + comb2._1, comb1._2 + comb2._2, comb1._3 + comb2._3)).collectAsMap()

      s = AddKeyTable.run(s, Array("-k", keyNames.map( k => k + " = " + "va.keys." + k).mkString(","),
        "-a", "nHet = gs.filter(g => g.isHet).count(), nCalled = gs.filter(g => g.isCalled).count(), nTotal = gs.count()",
        "-o", outputFile))

      val keyTableResults = getKeyTableResults(outputFile, keyNames)

      keyTableEqualAnnExpr(truthResult, keyTableResults)
    }

  property("aggregate by sample groups same") = forAll(compGenSample) { case (vds, phenotypes) =>
      val outputFile = tmpDir.createTempFile("aggBySampleGroup", ".tsv")

      val nPhenotypes = phenotypes.length
      val keyNames = (1 to nPhenotypes).map("key" + _)

      var signature = TStruct()
      keyNames.foreach(k => signature = signature.appendKey(k, TString))

      val phenoMap = vds.sampleIds.zipWithIndex.map{ case (sid, i) =>
        (sid, Annotation(phenotypes.map(_(i).orNull).toSeq : _*))
      }.toMap

      var s = State(sc, sqlContext, vds.annotateSamples(phenoMap, signature, "sa.pheno"))

      s = AnnotateSamplesExpr.run(s, Array("-c", "sa.nHet = gs.filter(g => g.isHet).count(), sa.nCalled = gs.filter(g => g.isCalled).count(), sa.nTotal = gs.count()"))

      val (_, nHetQuery) = s.vds.querySA("sa.nHet")
      val (_, nCalledQuery) = s.vds.querySA("sa.nCalled")
      val (_, nTotalQuery) = s.vds.querySA("sa.nTotal")
      val (_, phenoQuery) = s.vds.querySA("sa.pheno.*")

      val truthResult = sc.parallelize(s.vds.sampleIdsAndAnnotations).map{ case (sid, sa) =>
        (phenoQuery(sa).get.asInstanceOf[IndexedSeq[String]], (nHetQuery(sa).get.asInstanceOf[Long], nCalledQuery(sa).get.asInstanceOf[Long], nTotalQuery(sa).get.asInstanceOf[Long]))
      }.aggregateByKey((0L, 0L, 0L))((comb, counts) => (comb._1 + counts._1, comb._2 + counts._2, comb._3 + counts._3),
        (comb1, comb2) => (comb1._1 + comb2._1, comb1._2 + comb2._2, comb1._3 + comb2._3)).collectAsMap()

      s = AddKeyTable.run(s, Array("-k", keyNames.map( k => k + " = " + "sa.pheno." + k).mkString(","),
        "-a", "nHet = gs.filter(g => g.isHet).count(), nCalled = gs.filter(g => g.isCalled).count(), nTotal = gs.count()",
        "-o", outputFile))

      val keyTableResults = getKeyTableResults(outputFile, keyNames)

      keyTableEqualAnnExpr(truthResult, keyTableResults)
    }

    property("aggregate by sample and variants same") = forAll(compGen) { case (vds, sampleGroups, variantGroups) =>
      val outputFile = tmpDir.createTempFile("aggBySampleVariantGroup", ".tsv")

      val nKeys = sampleGroups.length + variantGroups.length
      val keyNames = (1 to nKeys).map("key" + _)
      val sampleKeyNames = (1 to sampleGroups.length).map("key" + _)
      val variantKeyNames = (sampleGroups.length + 1 to nKeys).map("key" + _)

      var sampleSignature = TStruct()
      sampleKeyNames.foreach(k => sampleSignature = sampleSignature.appendKey(k, TString))

      var variantSignature = TStruct()
      variantKeyNames.foreach(k => variantSignature = variantSignature.appendKey(k, TString))

      val sampleMap = vds.sampleIds.zipWithIndex.map{ case (sid, i) =>
        (sid, Annotation(sampleGroups.map(_(i).orNull).toSeq : _*))
      }.toMap

      val variantAnnotations = sc.parallelize(vds.variants.collect().zipWithIndex.map { case (v, i) =>
        (v, Annotation(variantGroups.map(_ (i).orNull).toSeq: _*))
      }).toOrderedRDD

      var s = State(sc, sqlContext, vds.annotateSamples(sampleMap, sampleSignature, "sa.keys")
        .annotateVariants(variantAnnotations, variantSignature, "va.keys"))

      val (_, sampleKeyQuery) = s.vds.querySA("sa.keys.*")
      val (_, variantKeyQuery) = s.vds.queryVA("va.keys.*")

      val keyGenotypeRDD = s.vds.mapWithAll{case (v, va, sid, sa, g) =>
        val key = sampleKeyQuery(sa).get.asInstanceOf[IndexedSeq[String]] ++ variantKeyQuery(va).get.asInstanceOf[IndexedSeq[String]]
        (key, g)
      }

      val result = keyGenotypeRDD.aggregateByKey((0L, 0L, 0L))(
        (comb, gt) => (comb._1 + gt.isHet.toInt.toInt, comb._2 + gt.isCalled.toInt.toInt, comb._3 + 1),
        (comb1, comb2) => (comb1._1 + comb2._1, comb1._2 + comb2._2, comb1._3 + comb2._3)).collectAsMap()

      s = AddKeyTable.run(s, Array("-k", (sampleKeyNames.map(k => k + " = " + "sa.keys." + k) ++ variantKeyNames.map(k => k + " = " + "va.keys." + k)).mkString(","),
        "-a", "nHet = gs.filter(g => g.isHet).count(), nCalled = gs.filter(g => g.isCalled).count(), nTotal = gs.count()",
        "-o", outputFile))

      val keyTableResults = getKeyTableResults(outputFile, keyNames)

      keyTableEqualAnnExpr(result, keyTableResults)
    }
  }

  @Test def testAddKeyTable() {
    Spec.check()
  }

}
