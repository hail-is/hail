package is.hail.io.gen

import is.hail.annotations._
import is.hail.expr.types._
import is.hail.io.bgen.BgenLoader
import is.hail.io.vcf.LoadVCF
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

import scala.collection.mutable

case class GenResult(file: String, nSamples: Int, nVariants: Int, rdd: RDD[(Annotation, Iterable[Annotation])])

object GenLoader {
  def apply(genFile: String, sampleFile: String, sc: SparkContext, gr: GenomeReference,
    nPartitions: Option[Int] = None, tolerance: Double = 0.02,
    chromosome: Option[String] = None, contigRecoding: Map[String, String] = Map.empty[String, String]): GenResult = {

    val hConf = sc.hadoopConfiguration
    val sampleIds = BgenLoader.readSampleFile(hConf, sampleFile)

    LoadVCF.warnDuplicates(sampleIds)

    val nSamples = sampleIds.length

    val rdd = sc.textFileLines(genFile, nPartitions.getOrElse(sc.defaultMinPartitions))
      .map(_.map { l =>
        readGenLine(l, nSamples, tolerance, gr, chromosome, contigRecoding)
      }.value)

    GenResult(genFile, nSamples, rdd.count().toInt, rdd = rdd)
  }

  def readGenLine(line: String, nSamples: Int,
    tolerance: Double,
    gr: GenomeReference,
    chromosome: Option[String] = None,
    contigRecoding: Map[String, String] = Map.empty[String, String]): (Annotation, Iterable[Annotation]) = {

    val arr = line.split("\\s+")
    val chrCol = if (chromosome.isDefined) 1 else 0
    val chr = chromosome.getOrElse(arr(0))
    val varid = arr(1 - chrCol)
    val rsid = arr(2 - chrCol)
    val start = arr(3 - chrCol)
    val ref = arr(4 - chrCol)
    val alt = arr(5 - chrCol)

    val recodedContig = contigRecoding.getOrElse(chr, chr)
    val variant = Variant(recodedContig, start.toInt, ref, alt, gr)

    val nGenotypes = 3
    val gp = arr.drop(6 - chrCol).map {
      _.toDouble
    }

    if (gp.length != (3 * nSamples))
      fatal("Number of genotype probabilities does not match 3 * number of samples. If no chromosome column is included, use -c to input the chromosome.")

    val gsb = new ArrayBuilder[Annotation]()

    for (i <- gp.indices by 3) {
      val d0 = gp(i)
      val d1 = gp(i + 1)
      val d2 = gp(i + 2)
      val sumDosages = d0 + d1 + d2

      val a =
        if (math.abs(sumDosages - 1.0) <= tolerance) {
          val gp = Array(d0 / sumDosages, d1 / sumDosages, d2 / sumDosages)
          val gt = Genotype.unboxedGTFromLinear(gp)
          Annotation(gt, gp: IndexedSeq[Double])
        } else
          null

      gsb += a
    }

    val annotations = Annotation(variant, rsid, varid)

    (annotations, gsb.result())
  }
}
