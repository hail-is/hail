package is.hail.io.gen

import is.hail.annotations._
import is.hail.expr._
import is.hail.io.bgen.BgenLoader
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.rdd.RDD
import org.apache.spark.{Accumulable, SparkContext}

import scala.collection.mutable

case class GenResult(file: String, nSamples: Int, nVariants: Int, rdd: RDD[(Variant, (Annotation, Iterable[Genotype]))])

object GenReport {
  final val gpNoCall = 0
  final val gpSumLessThanTolerance = 1
  final val gpSumGreaterThanTolerance = 2

  var accumulators: List[(String, Accumulable[mutable.Map[Int, Int], Int])] = Nil

  def warningMessage(id: Int, count: Int): String = {
    val desc = (id: @unchecked) match {
      case `gpNoCall` => "Genotype probabilities are (0.0,0.0,0.0)"
      case `gpSumLessThanTolerance` => "Sum of genotype probabilities < (1.0 - tolerance)"
      case `gpSumGreaterThanTolerance` => "Sum of genotype probabilities > (1.0 + tolerance)"
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

object GenLoader {
  def apply(genFile: String, sampleFile: String, sc: SparkContext,
    nPartitions: Option[Int] = None, tolerance: Double = 0.02,
    chromosome: Option[String] = None): GenResult = {

    val hConf = sc.hadoopConfiguration
    val sampleIds = BgenLoader.readSampleFile(hConf, sampleFile)

    if (sampleIds.length != sampleIds.toSet.size)
      fatal(s"Duplicate sample IDs exist in $sampleFile")

    val nSamples = sampleIds.length

    val reportAcc = sc.accumulable[mutable.Map[Int, Int], Int](mutable.Map.empty[Int, Int])
    GenReport.accumulators ::= (genFile, reportAcc)

    val rdd = sc.textFileLines(genFile, nPartitions.getOrElse(sc.defaultMinPartitions))
      .map(_.map { l =>
        readGenLine(l, nSamples, tolerance, reportAcc, chromosome)
      }.value)

    val signatures = TStruct("rsid" -> TString, "varid" -> TString)

    GenResult(genFile, nSamples, rdd.count().toInt, rdd = rdd)
  }

  def readGenLine(line: String, nSamples: Int,
    tolerance: Double,
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
    val gp = arr.drop(6 - chrCol).map {
      _.toDouble
    }

    if (gp.length != (3 * nSamples))
      fatal("Number of genotype probabilities does not match 3 * number of samples. If no chromosome column is included, use -c to input the chromosome.")

    val gsb = new GenotypeStreamBuilder(2, isLinearScale = true)
    val gb = new GenotypeBuilder(2, isLinearScale = true)

    for (i <- gp.indices by 3) {
      gb.clear()

      val d0 = gp(i)
      val d1 = gp(i + 1)
      val d2 = gp(i + 2)
      val sumDosages = d0 + d1 + d2
      if (sumDosages == 0.0)
        reportAcc += GenReport.gpNoCall
      else if (math.abs(sumDosages - 1.0) > tolerance)
        reportAcc += GenReport.gpSumLessThanTolerance
      else {
        val px = Genotype.weightsToLinear(d0, d1, d2)
        val gt = Genotype.gtFromLinear(px)

        gt.foreach(gt => gb.setGT(gt))
        gb.setPX(px)
      }

      gsb.write(gb)
    }

    val annotations = Annotation(rsid, varid)

    (variant, (annotations, gsb.result()))
  }
}