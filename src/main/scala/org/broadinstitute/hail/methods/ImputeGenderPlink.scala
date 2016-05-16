package org.broadinstitute.hail.methods

import java.nio.ByteBuffer

import org.apache.spark.SparkEnv
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant.{VariantDataset, Variant, Genotype}
import org.broadinstitute.hail.annotations._

object ImputeGenderPlink {
  def imputeGender(vds: VariantDataset) = new ImputeGenderPlink(vds)
  def imputeGender(vds: VariantDataset, mafThreshold: Double, excludePAR: Boolean) =
    new ImputeGenderPlink(vds, mafThreshold, excludePAR)

  def determineSex(ibc:InbreedingCombiner): Option[Int] = {
    ibc.F match {
      case Some(x) =>
        if (x <= 0.2)
          Option(2)
        else if (x >= 0.8)
          Option(1)
        else
          None
      case None => None
    }
  }
}

class ImputeGenderPlink(vds: VariantDataset, mafThreshold: Double = 0.0,
                        excludePAR: Boolean = true) {

  import ImputeGenderPlink._

  private def xChrVds =  {
    if (excludePAR)
      vds.filterVariants(
      (v: Variant, va: Annotation, gs: Iterable[Genotype]) => !v.inParX && (v.contig == "X" || v.contig == "23")
    )
    else
      vds.filterVariants(
        (v: Variant, va: Annotation, gs: Iterable[Genotype]) => v.contig == "X" || v.contig == "23"
    )
  }

  private def populationParameters = xChrVds.rdd.map { case (v, va, gs) =>
    val nCalled = gs.map { g => if (g.isCalled) 1 else 0 }.sum
    val nAltAlleles = gs.map { g => if (g.isHet) 1 else if (g.isHomVar) 2 else 0 }.sum
    val maf: Option[Double] = divOption(nAltAlleles, 2 * nCalled)
    (v, (maf, nCalled))
  }

  private def filteredXchrRDD = {
    val localMafThreshold = mafThreshold

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
  }

  private def inbreedingCoefficients: RDD[(String, InbreedingCombiner)] = {
    val sparkContext = xChrVds.sparkContext
    val sampleIdsBC = xChrVds.sampleIds

    val serializer = SparkEnv.get.serializer.newInstance()
    val zeroBuffer = serializer.serialize(new InbreedingCombiner)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    filteredXchrRDD
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

  def result = inbreedingCoefficients.map { case (s, ibc) => (s, Annotation(ibc.F, ibc.E, ibc.O, ibc.N, ibc.T, determineSex(ibc))) }
}





