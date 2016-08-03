package org.broadinstitute.hail.io.gen

import org.apache.spark.rdd.RDD
import org.apache.spark.{Accumulable, SparkContext}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.bgen.BgenLoader
import org.broadinstitute.hail.io.gen.GenUtils._
import org.broadinstitute.hail.variant._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

case class GenResult(file: String, nSamples: Int, nVariants: Int, rdd: RDD[(Variant, (Annotation, Iterable[Genotype]))])

object GenReport {
  final val dosageNoCall = 0
  final val dosageLessThanTolerance = 1
  final val dosageGreaterThanTolerance = 2

  var accumulators: List[(String, Accumulable[mutable.Map[Int, Int], Int])] = Nil

  def warningMessage(id: Int, count: Int): String = {
    val desc = (id: @unchecked) match {
      case `dosageNoCall` => "Dosage triple of (0.0,0.0,0.0)"
      case `dosageLessThanTolerance` => "Sum of Dosage < (1.0 - tolerance)"
      case `dosageGreaterThanTolerance` => "Sum of Dosage > (1.0 + tolerance)"
    }
    s"$count ${ plural(count, "time") }: $desc"
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
    if (sum == 0d)
      fatal("found invalid set of PP values that sum to zero")
    else
      arr.map(_ / sum)
  }

  def convertProbToInt(prob: Double): Int = {
    val tmp = prob * 32768
    require(tmp >= 0d && tmp < 65535.5)
    math.round(tmp).toInt
  }

  lazy val phredConversionTable: Array[Double] = (0 to 65535).map { i => -10 * math.log10(if (i == 0) .25 else i) }.toArray
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

    val rdd = sc.textFileLines(genFile, nPartitions.getOrElse(sc.defaultMinPartitions))
      .map(_.map { l =>
        readGenLine(l, nSamples, tolerance, compress, reportAcc, chromosome)
      }.value)

    val signatures = TStruct("rsid" -> TString, "varid" -> TString)

    GenResult(genFile, nSamples, rdd.count().toInt, rdd = rdd)
  }

  def readGenLine(line: String, nSamples: Int,
                  tolerance: Double, compress: Boolean,
                  reportAcc: Accumulable[mutable.Map[Int, Int], Int],
                  chromosome: Option[String] = None): (Variant, (Annotation, Iterable[Genotype])) = {

    val arr = line.split("\\s+")
    val chrCol = if (chromosome.isDefined) 1 else 0
    val chr = chromosome.getOrElse(arr(0))
    val varid = arr(1 - chrCol)
    val rsid = arr(2 - chrCol)
    val start = arr(3 - chrCol)
    val ref = arr(4 - chrCol)
    val alt = arr(5 - chrCol)

    val recodedChr = chr match {
      case "23" => "X"
      case "24" => "Y"
      case "25" => "X"
      case "26" => "MT"
      case x => x
    }

    val variant = Variant(recodedChr, start.toInt, ref, alt)
    val nGenotypes = 3
    val dosages = arr.drop(6 - chrCol).map {
      _.toDouble
    }

    if (dosages.length != (3 * nSamples))
      fatal("Number of dosages does not match number of samples. If no chromosome column is included, use -c to input the chromosome.")

    val dosageArray = new Array[Int](3)
    val b = new GenotypeStreamBuilder(2, isDosage = true, compress)
    val genoBuilder = new GenotypeBuilder(2, isDosage = true)

    for (i <- dosages.indices by 3) {
      genoBuilder.clear()

      val origDosages = Array(dosages(i), dosages(i + 1), dosages(i + 2))
      val sumDosages = origDosages.sum

      if (sumDosages == 0.0)
        reportAcc += GenReport.dosageNoCall
      else if (math.abs(sumDosages - 1.0) > tolerance)
        reportAcc += GenReport.dosageLessThanTolerance
      else {
        val normIntDosages = normalizePPs(origDosages).map(convertProbToInt)
        val sumIntDosages = normIntDosages.sum
        assert(sumIntDosages >= 32768 - nGenotypes && sumIntDosages <= 32768 + nGenotypes)

        val maxIntDosage = normIntDosages.max
        val gt = {
          if (maxIntDosage < 16384 && normIntDosages.count(_ == maxIntDosage) != 1) //first comparison for speed to not evaluate count if prob > 0.5
            -1
          else
            normIntDosages.indexOf(maxIntDosage)
        }

        if (gt >= 0) {
          genoBuilder.setGT(gt)
        }

        genoBuilder.setPX(normIntDosages)
      }

      b.write(genoBuilder)
    }

    val annotations = Annotation(rsid, varid)

    (variant, (annotations, b.result()))
  }
}