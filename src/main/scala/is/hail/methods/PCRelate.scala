package is.hail.methods

import org.apache.spark.rdd.RDD
import is.hail.utils._
import is.hail.keytable.KeyTable
import is.hail.variant.{Genotype, Variant, VariantDataset}
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.distributed._

import scala.collection.generic.CanBuildFrom
import scala.language.higherKinds
import scala.reflect.ClassTag

object PCRelate {

  /**
    *
    * @param vds
    * @param mean (variant, (sample, mean))
    * @return
    */
  def apply(vds: VariantDataset, mean: RDD[(Variant, (Int, Double))]): RDD[((String, String), Double)] = {
    assert(vds.wasSplit, "PCRelate requires biallelic VDSes")

    // (variant, (sample, gt))
    val g = vds.rdd.flatMap { case (v, (va, gs)) =>
      gs.zipWithIndex.map { case (g, i) =>
        (v, (i, g.nNonRefAlleles.getOrElse[Int](-1): Double)) } }

    val meanPairs = mean.join(mean)
      .filter { case (_, ((i, _), (j, _))) => j >= i }
      .map { case (vi, ((s1, mean1), (s2, mean2))) =>
        ((s1, s2, vi), (mean1, mean2))
    }

    val numerator = g.join(g)
      .filter { case (_, ((i, _), (j, _))) => j >= i }
      .map { case (vi, ((s1, gt1), (s2, gt2))) =>
        ((s1, s2, vi), (gt1, gt2))
    }
      .join(meanPairs)
      .map { case ((s1, s2, vi), ((gt1, gt2), (mean1, mean2))) =>
        ((s1, s2), (gt1 - 2 * mean1) * (gt2 - 2 * mean2))
    }
      .reduceByKey(_ + _)

    val denominator = mean.join(mean)
      .filter { case (_, ((i, _), (j, _))) => j >= i }
      .map { case (vi, ((s1, mean1), (s2, mean2))) =>
        ((s1, s2), Math.sqrt(mean1 * (1 - mean1) * mean2 * (1 - mean2)))
    }
      .reduceByKey(_ + _)

    val sampleIndexToId =
      vds.sampleIds.zipWithIndex.map { case (s, i) => (i, s) }.toMap

    numerator.join(denominator)
      .map { case ((s1, s2), (numerator, denominator)) => ((s1, s2), numerator / denominator / 4) }
      .map { case ((s1, s2), x) => ((sampleIndexToId(s1), sampleIndexToId(s2)), x) }
  }

  /**
    *
    * @param pcs [nSample, nPCs], sample major
    * @return
    */
  def mean(vds: VariantDataset, pcs: Array[Array[Double]]): RDD[(Variant, (Int, Double))] = {
    val pcsbc = vds.sparkContext.broadcast(pcs)
    vds.rdd.flatMap { case (v, (va, gs)) =>
      val ols = new org.apache.commons.math3.stat.regression.OLSMultipleLinearRegression()
      val y = gs.map(g => g.nNonRefAlleles.getOrElse(0) + 0.0).toArray
      ols.newSampleData(y, pcsbc.value)
	    val b = ols.estimateRegressionParameters()
      val b0 = ols.estimateResiduals()
      pcs.zip(b0).zipWithIndex.map { case ((pcs, b0), i) =>
        (v, (i, b0 + b.zip(pcs).map { case (x,y) => x*y }.sum: Double)) }
    }
  }

}
