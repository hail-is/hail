package is.hail.methods

import is.hail.SparkSuite
import is.hail.check.{Gen, Prop}
import is.hail.utils._
import is.hail.variant.{VSMSubgen, Variant, VariantSampleMatrix}
import org.apache.spark.SparkContext
import org.testng.annotations.Test

import scala.language._


class ConcordanceSuite extends SparkSuite {
  def gen(sc: SparkContext) = for (vds1 <- VariantSampleMatrix.gen(hc, VSMSubgen.plinkSafeBiallelic);
    vds2 <- VariantSampleMatrix.gen(hc, VSMSubgen.plinkSafeBiallelic);
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

  //FIXME use SnpSift when it's fixed
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

  //FIXME use SnpSift when it's fixed
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

  @Test def testCombiner() {
    val comb = new ConcordanceCombiner

    comb.mergeBoth(-1, 1)
    comb.mergeBoth(-1, 1)
    comb.mergeBoth(-1, 1)
    comb.mergeBoth(-1, 1)
    comb.mergeRight(2)
    comb.mergeLeft(0)

    assert(comb.toAnnotation == IndexedSeq(
      IndexedSeq(0L, 0L, 0L, 0L, 1L),
      IndexedSeq(0L, 0L, 0L, 4L, 0L),
      IndexedSeq(1L, 0L, 0L, 0L, 0L),
      IndexedSeq(0L, 0L, 0L, 0L, 0L),
      IndexedSeq(0L, 0L, 0L, 0L, 0L)
    ))

    val comb2 = new ConcordanceCombiner

    comb2.mergeLeft(2)
    comb2.mergeLeft(2)
    comb2.mergeLeft(-1)
    comb2.mergeLeft(-1)
    comb2.mergeLeft(2)
    comb2.mergeRight(0)
    comb2.mergeRight(1)
    comb2.mergeRight(1)
    comb2.mergeBoth(1, 1)
    comb2.mergeBoth(1, 1)
    comb2.mergeBoth(-1, 1)
    comb2.mergeBoth(-1, 1)
    comb2.mergeBoth(1, -1)
    comb2.mergeBoth(1, -1)
    comb2.mergeBoth(2, -1)
    comb2.mergeBoth(2, -1)
    comb2.mergeBoth(2, -1)

    assert(comb2.toAnnotation == IndexedSeq(
      IndexedSeq(0L, 0L, 1L, 2L, 0L),
      IndexedSeq(2L, 0L, 0L, 2L, 0L),
      IndexedSeq(0L, 0L, 0L, 0L, 0L),
      IndexedSeq(0L, 2L, 0L, 2L, 0L),
      IndexedSeq(3L, 3L, 0L, 0L, 0L)
    ))
  }

  @Test def test() {
    Prop.forAll(gen(sc).filter { case (vds1, vds2) =>
      vds1.sampleIds.toSet.intersect(vds2.sampleIds.toSet).nonEmpty &&
        vds1.variants.intersection(vds2.variants).count() > 0
    }) { case (vds1, vds2) =>

      val variants1 = vds1.variants.collect().toSet
      val variants2 = vds2.variants.collect().toSet

      val commonVariants = variants1.intersect(variants2)

      val uniqueVds1Variants = (variants1 -- commonVariants).size
      val uniqueVds2Variants = (variants2 -- commonVariants).size

      val innerJoin = vds1.expand().map { case (v, s, g) => ((v, s), g) }
        .join(vds2.expand().map { case (v, s, g) => ((v, s), g) })

      val innerJoinSamples = innerJoin.map { case (k, v) => (k._2, v) }
        .aggregateByKey(new ConcordanceCombiner)({ case (comb, (g1, g2)) =>
          comb.mergeBoth(g1.unboxedGT, g2.unboxedGT)
          comb
        }, { case (comb1, comb2) => comb1.merge(comb2) })
        .map { case (s, comb) => (s, comb.toAnnotation.tail.map(_.tail)) }
        .collectAsMap

      val innerJoinVariants = innerJoin.map { case (k, v) => (k._1, v) }
        .aggregateByKey(new ConcordanceCombiner)({ case (comb, (g1, g2)) =>
          comb.mergeBoth(g1.unboxedGT, g2.unboxedGT)
          comb
        }, { case (comb1, comb2) => comb1.merge(comb2) })
        .collectAsMap
        .mapValues(_.toAnnotation)

      val (globals, samples, variants) = vds1.concordance(vds2)

      val (_, queryUnique1Sum) = samples.querySA("sa.concordance[0].sum()")
      val (_, queryUnique2Sum) = samples.querySA("sa.concordance.map(x => x[0]).sum()")
      val (_, innerJoinQuery) = samples.querySA("sa.concordance.map(x => x[1:])[1:]")

      samples.sampleIdsAndAnnotations.foreach { case (s, sa) =>
        assert(queryUnique1Sum(sa) == uniqueVds2Variants)
        assert(queryUnique2Sum(sa) == uniqueVds1Variants)
        assert(innerJoinQuery(sa) == innerJoinSamples(s))
      }

      val (_, variantQuery) = variants.queryVA("va.concordance")
      variants.variantsAndAnnotations.collect()
        .foreach { case (v, va) =>
          innerJoinVariants.get(v).forall(variantQuery(va) == _)
        }
      true
    }.check()
  }
}
