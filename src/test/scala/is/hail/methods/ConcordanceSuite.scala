package is.hail.methods

import is.hail.SparkSuite
import is.hail.annotations.Annotation
import is.hail.check.{Gen, Prop}
import is.hail.expr.types.{TString, TStruct, TVariant}
import is.hail.table.Table
import is.hail.utils._
import is.hail.testUtils._
import is.hail.variant._
import org.apache.spark.SparkContext
import org.apache.spark.sql.Row
import org.testng.annotations.Test

import scala.language._

class ConcordanceSuite extends SparkSuite {
  def gen(sc: SparkContext) = for {
    gr <- GenomeReference.gen
    vds1 <- MatrixTable.gen(hc, VSMSubgen.plinkSafeBiallelic.copy(vSigGen = Gen.const(TVariant(gr))))
    vds2 <- MatrixTable.gen(hc, VSMSubgen.plinkSafeBiallelic.copy(vSigGen = Gen.const(TVariant(gr))))
    scrambledIds1 <- Gen.shuffle(vds1.stringSampleIds).map(_.iterator)
    newIds2 <- Gen.parameterized { p =>
      Gen.const(vds2.stringSampleIds.map { id =>
        if (scrambledIds1.hasNext && p.rng.nextUniform(0, 1) < .5) {
          val newId = scrambledIds1.next()
          if (!vds2.stringSampleIds.contains(newId)) newId else id
        }
        else
          id
      })
    }
    scrambledVariants1 <- Gen.shuffle(vds1.variants.collect()).map(_.iterator)
    newVariantMapping <- Gen.parameterized { p =>
      Gen.const(Table.parallelize(hc, vds2.variants.collect().map { v =>
        if (scrambledVariants1.hasNext && p.rng.nextUniform(0, 1) < .5) Row(v, scrambledVariants1.next()) else Row(v, v)
      }, TStruct("v1" -> TVariant(gr),
        "v2" -> TVariant(gr)), Array.empty[String], None)
        .annotate("locus = v1.locus, alleles = [v1.ref].extend(v1.altAlleles.map(x => x.alt))")
        .keyBy("locus", "alleles"))
    }
  } yield (vds1, vds2.annotateVariantsTable(newVariantMapping, "newVariant")
      .annotateVariantsExpr("locus = va.newVariant.v2.locus, " +
        "alleles = [va.newVariant.v2.ref].extend(va.newVariant.v2.altAlleles.map(x => x.alt))")
      .copy2(colValues = newIds2.map(Annotation(_)), colType = TStruct("s" -> TString())))

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

    comb.merge(1, 3)
    comb.merge(1, 3)
    comb.merge(1, 3)
    comb.merge(1, 3)
    comb.merge(0, 4)
    comb.merge(2, 0)

    assert(comb.toAnnotation == IndexedSeq(
      IndexedSeq(0L, 0L, 0L, 0L, 1L),
      IndexedSeq(0L, 0L, 0L, 4L, 0L),
      IndexedSeq(1L, 0L, 0L, 0L, 0L),
      IndexedSeq(0L, 0L, 0L, 0L, 0L),
      IndexedSeq(0L, 0L, 0L, 0L, 0L)
    ))

    val comb2 = new ConcordanceCombiner

    comb2.merge(4, 0)
    comb2.merge(4, 0)
    comb2.merge(1, 0)
    comb2.merge(1, 0)
    comb2.merge(4, 0)
    comb2.merge(0, 2)
    comb2.merge(0, 3)
    comb2.merge(0, 3)
    comb2.merge(3, 3)
    comb2.merge(3, 3)
    comb2.merge(1, 3)
    comb2.merge(1, 3)
    comb2.merge(3, 1)
    comb2.merge(3, 1)
    comb2.merge(4, 1)
    comb2.merge(4, 1)
    comb2.merge(4, 1)

    assert(comb2.toAnnotation == IndexedSeq(
      IndexedSeq(0L, 0L, 1L, 2L, 0L),
      IndexedSeq(2L, 0L, 0L, 2L, 0L),
      IndexedSeq(0L, 0L, 0L, 0L, 0L),
      IndexedSeq(0L, 2L, 0L, 2L, 0L),
      IndexedSeq(3L, 3L, 0L, 0L, 0L)
    ))

    assert(comb2.nDiscordant == 0)
  }

  @Test def testNDiscordant() {
    val g = (for {i <- Gen.choose(-2, 2)
      j <- Gen.choose(-2, 2)} yield (i, j)).filter { case (i, j) => !(i == -2 && j == -2) }
    val seqG = Gen.buildableOf[Array](g)

    val comb = new ConcordanceCombiner

    Prop.forAll(seqG) { values =>
      comb.reset()

      var n = 0
      values.foreach { case (i, j) =>
        if (i == -2)
          comb.merge(0, j + 2)
        else if (j == -2)
          comb.merge(i + 2, 0)
        else {
          if (i >= 0 && j >= 0 && i != j)
            n += 1
          comb.merge(i + 2, j + 2)
        }
      }
      n == comb.nDiscordant
    }.check()
  }

  @Test def test() {
    Prop.forAll(gen(sc).filter { case (vds1, vds2) =>
      vds1.stringSampleIds.toSet.intersect(vds2.stringSampleIds.toSet).nonEmpty &&
        vds1.variants.intersection(vds2.variants).count() > 0
    }) { case (vds1, vds2) =>

      val variants1 = vds1.variants.collect().toSet
      val variants2 = vds2.variants.collect().toSet

      val commonVariants = variants1.intersect(variants2)

      val uniqueVds1Variants = (variants1 -- commonVariants).size
      val uniqueVds2Variants = (variants2 -- commonVariants).size

      val innerJoin = vds1.expand().map { case (v, s, g) => ((v, s), g) }
        .join(vds2.expand().map { case (v, s, g) => ((v, s), g) })

      def getIndex(g: Annotation): Int = {
        Genotype.call(g)
          .map(Call.nNonRefAlleles)
          .getOrElse(-1)
      }

      val innerJoinSamples = innerJoin.map { case (k, v) => (k._2, v) }
        .aggregateByKey(new ConcordanceCombiner)({ case (comb, (g1, g2)) =>
          comb.merge(getIndex(g1) + 2, getIndex(g2) + 2)
          comb
        }, { case (comb1, comb2) => comb1.merge(comb2) })
        .map { case (s, comb) => (s, comb.toAnnotation.tail.map(_.tail)) }
        .collectAsMap

      val innerJoinVariants = innerJoin.map { case (k, v) => (k._1, v) }
        .aggregateByKey(new ConcordanceCombiner)({ case (comb, (g1, g2)) =>
          comb.merge(getIndex(g1) + 2, getIndex(g2) + 2)
          comb
        }, { case (comb1, comb2) => comb1.merge(comb2) })
        .collectAsMap
        .mapValues(_.toAnnotation)

      val (globals, samples, variants) = CalculateConcordance(vds1, vds2)

      samples.rdd.collect().foreach { r =>
        assert(r.getAs[IndexedSeq[IndexedSeq[Long]]](2).apply(0).sum == uniqueVds2Variants)
        assert(r.getAs[IndexedSeq[IndexedSeq[Long]]](2).map(_.apply(0)).sum == uniqueVds1Variants)
        assert(r.getAs[IndexedSeq[IndexedSeq[Long]]](2).map(_.tail).tail == innerJoinSamples(r.getAs[String](0)))
      }

      variants.rdd.collect()
        .foreach { r =>
          assert(innerJoinVariants.get(r.getAs[Variant](0)).forall(r.get(2) == _))
        }
      true
    }.check()
  }
}
