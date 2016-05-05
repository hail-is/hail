package org.broadinstitute.hail.io.gen

import org.apache.spark.rdd.RDD
import org.apache.spark.{Accumulable, SparkContext}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.bgen.BgenLoader
import org.broadinstitute.hail.variant._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

case class GenResult(file: String, nSamples: Int, nVariants: Int, rdd: RDD[(Variant, Annotation, Iterable[Genotype])])

object GenReport {
  val dosageNoCall = 0
  val dosageLessThanTolerance = 1

  var accumulators: List[(String, Accumulable[mutable.Map[Int, Int], Int])] = Nil

  def warningMessage(id: Int, count: Int): String = {
    val desc = id match {
      case `dosageNoCall` => "Dosage triple of (0.0,0.0,0.0)"
      case `dosageLessThanTolerance` => "Sum of Dosage < (1.0 - tolerance)"
      case _ => throw new UnsupportedOperationException
    }
    s"$count ${plural(count, "time")}: $desc"
  }

  def report() {
    val sb = new StringBuilder()

    for ((file, m) <- accumulators) {
      sb.clear()

      sb.append(s"while importing:\n    $file")

      val genotypeWarnings = m.value
      val nGenotypesFiltered = genotypeWarnings.values.sum
      if (nGenotypesFiltered > 0) {
        sb.append(s"\n  filtered $nGenotypesFiltered genotypes:")
        genotypeWarnings.foreach { case (id, n) =>
          if (n > 0) {
            sb.append("\n    ")
            sb.append(warningMessage(id, n))
          }
        }
      }

      if (nGenotypesFiltered == 0) {
        sb.append("  import clean")
        info(sb.result())
      } else
        warn(sb.result())
    }
  }
}

object GenUtils {
  def normalizePPs(arr: Array[Double]): Array[Double] = {
    val sum = arr.sum
    if (sum != 0.0)
      if (math.abs(sum - 1.0) > 3.0e-4)
        arr.map{_ / sum}
      else
        arr
    else
      Array(0.3333, 0.3333, 0.3333)
  }

  def convertProbsToInt(prob: Double): Int = {
    val tmp = prob * 32768
    require(tmp >= 0 && tmp < 65535.5)
    math.round(tmp).toInt
  }

  def convertProbsToInt(probArray: Array[Double]): Array[Int] = probArray.map{ d => convertProbsToInt(d)}

}

object GenLoader {
  def apply(genFile: String, sampleFile: String, sc: SparkContext,
            nPartitions: Option[Int] = None, tolerance: Double = 0.02,
            compress: Boolean = false, chromosome: Option[String] = None): GenResult = {

    val hConf = sc.hadoopConfiguration
    val sampleIds = BgenLoader.readSampleFile(hConf, sampleFile)

    if (sampleIds.length != sampleIds.toSet.size)
      fatal(s"Duplicate sample IDs exist in $sampleFile")

    val nSamples = sampleIds.length

    val reportAcc = sc.accumulable[mutable.Map[Int, Int], Int](mutable.Map.empty[Int, Int])
    GenReport.accumulators ::=(genFile, reportAcc)

    val rdd = sc.textFile(genFile, nPartitions.getOrElse(sc.defaultMinPartitions))
        .map{ case line => readGenLine(line, nSamples, tolerance, compress, chromosome, reportAcc)}

    val signatures = TStruct("rsid" -> TString, "varid" -> TString)

    GenResult(genFile, nSamples, rdd.count().toInt, rdd = rdd)
  }

  def readGenLine(line: String, nSamples: Int,
                  tolerance: Double, compress: Boolean,
                  chromosome: Option[String] = None, reportAcc: Accumulable[mutable.Map[Int, Int], Int]): (Variant, Annotation, Iterable[Genotype]) = {
    val arr = line.split("\\s+")
    val chrCol = if (chromosome.isDefined) 1 else 0
    val chr = chromosome.getOrElse(arr(0))
    val varid = arr(1 - chrCol)
    val rsid = arr(2 - chrCol)
    val start = arr(3 - chrCol)
    val ref = arr(4 - chrCol)
    val alt = arr(5 - chrCol)

    val variant = Variant(chr, start.toInt, ref, alt)
    val dosages = arr.drop(6 - chrCol).map {_.toDouble}

    if (dosages.length != (3 * nSamples))
      fatal("Number of dosages does not match number of samples. If no chromosome is given, make sure you use -c to input the chromosome.")

    val dosageArray = new Array[Int](3)
    val b = new GenotypeStreamBuilder(variant, compress)
    val genoBuilder = new GenotypeBuilder(variant)

    for (i <- dosages.indices by 3) {
      genoBuilder.clear()
      genoBuilder.setDosageFlag()

      val origDosages = Array(dosages(i), dosages(i+1), dosages(i+2))
      val sumDosages = origDosages.sum

      if (sumDosages == 0.0)
        reportAcc += GenReport.dosageNoCall
      else if (math.abs(sumDosages - 1.0) > tolerance)
        reportAcc += GenReport.dosageLessThanTolerance
      else {
        val normProbs = GenUtils.normalizePPs(origDosages)

        val dosageAA = GenUtils.convertProbsToInt(normProbs(0))
        val dosageAB = GenUtils.convertProbsToInt(normProbs(1))
        val dosageBB = GenUtils.convertProbsToInt(normProbs(2))

        val sumDosage = dosageAA + dosageAB + dosageBB

        assert(sumDosage >= 32768 - variant.nGenotypes && sumDosage <= 32768 + variant.nGenotypes)

        val gt = if (dosageAA > dosageAB && dosageAA > dosageBB)
          0
        else if (dosageAB > dosageAA && dosageAB > dosageBB)
          1
        else if (dosageBB > dosageAA && dosageBB > dosageAB)
          2
        else
          -1

        if (gt >= 0) {
          genoBuilder.setGT(gt)
        }

        genoBuilder.setDosage(Array(dosageAA, dosageAB, dosageBB))

      }
      b.write(genoBuilder)
    }

    val annotations = Annotation(rsid, varid)

    (variant, annotations, b.result())
  }
}