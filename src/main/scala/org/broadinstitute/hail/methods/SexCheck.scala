package org.broadinstitute.hail.methods

import java.nio.ByteBuffer

import org.apache.spark.SparkEnv
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant.{VariantDataset, Variant, Genotype}
import org.broadinstitute.hail.annotations._

object SexCheckPlink {
  def calcSex(vds: VariantDataset) = new SexCheckPlink(vds)
  def calcSex(vds: VariantDataset, mafThreshold: Double) = new SexCheckPlink(vds, mafThreshold)

  def determineSex(ibc:InbreedingCombiner):Int = {
    if (ibc.N < 1000) // Jackie addition
      0
    else if (ibc.F <= 0.2) //Female
      2
    else if (ibc.F >= 0.8) //Male
      1
    else
      0
  }
}

class SexCheckPlink(vds: VariantDataset, mafThreshold: Double = 1e-6) {

  import SexCheckPlink._

  def xChrVds = vds.filterVariants(
    (v: Variant, va: Annotation, gs: Iterable[Genotype]) => !v.inParX && (v.contig == "X" || v.contig == "23")
  )

  def yChrVds = vds.filterVariants(
    (v: Variant, va: Annotation, gs: Iterable[Genotype]) => !v.inParY && (v.contig == "Y" || v.contig == "24")
  )

  def populationParameters = xChrVds.rdd.map { case (v, va, gs) =>
    val nCalled = gs.map { g => if (g.isCalled) 1 else 0 }.sum
    val nAltAlleles = gs.map { g => if (g.isHet) 1 else if (g.isHomVar) 2 else 0 }.sum
    val maf: Option[Double] = divOption(nAltAlleles, 2 * nCalled)
    (v, (maf, nCalled))
  }

  def inbreedingCoefficients: RDD[(String, InbreedingCombiner)] = {
    val sparkContext = xChrVds.sparkContext
    val localMafThreshold = mafThreshold
    val sampleIdsBC = xChrVds.sampleIds

    val serializer = SparkEnv.get.serializer.newInstance()
    val zeroBuffer = serializer.serialize(new InbreedingCombiner)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    xChrVds.rdd
      .zipPartitions(populationParameters) { case (it, jt) =>
        it.zip(jt).map { case ((v, va, gs), (v2, vinfo)) =>
          require(v == v2)
          (v, va, gs, vinfo)
        }
      }

      .filter { case (v, va, gs, (maf, nCalled)) =>
        maf.isDefined && maf.get >= localMafThreshold && nCalled > 0
      }

      .map { case (v, va, gs, (maf, nCalled)) => (v, va, gs, (maf.get, nCalled))}

      .mapPartitions { (it: Iterator[(Variant, Annotation, Iterable[Genotype], Any)]) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        def copyZeroValue() = serializer.deserialize[InbreedingCombiner](ByteBuffer.wrap(zeroArray))
        val arrayZeroValue = Array.fill[InbreedingCombiner](sampleIdsBC.length)(copyZeroValue())

        sampleIdsBC.iterator
          .zip(it.foldLeft(arrayZeroValue) { case (acc, (v, va, gs, (maf: Double, nCalled: Int))) =>
            for ((g, i) <- gs.iterator.zipWithIndex)
              acc(i) = acc(i).addCount(g, maf, nCalled)
            acc
          }.iterator)
      }.foldByKey(new InbreedingCombiner)((ibc1, ibc2) => ibc1.combineCounts(ibc2))

    .map { case (s, ibc) => (s,ibc)}
  }

  def imputedSex = inbreedingCoefficients.map { case (i, ibc) => (i, determineSex(ibc)) }

}





