package org.broadinstitute.hail.io

import org.apache.hadoop.conf.Configuration
import org.apache.spark.storage.StorageLevel
import org.apache.spark.{RangePartitioner, SparkContext}
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant._


object GenLoader {
  def apply(genFile: String, sampleFile: String, sc: SparkContext, nPartitions: Option[Int] = None): VariantSampleMatrix[Genotype] = {
    val hConf = sc.hadoopConfiguration
    val sampleIds = BgenLoader.readSampleFile(hConf, sampleFile)
    val nSamples = sampleIds.length
    val rdd = sc.textFile(genFile, nPartitions.getOrElse(sc.defaultMinPartitions)).map{case line => readGenLine(line, nSamples)}
    val signatures = TStruct("rsid" -> TString, "lid" -> TString)
    VariantSampleMatrix(metadata = VariantMetadata(sampleIds).copy(vaSignature = signatures, wasSplit = true), rdd = rdd)
  }

  def convertPPsToInt(prob: Double): Int = {
    val tmp = prob * 32768
    require(tmp >= 0 && tmp < 65535.5)
    math.round(tmp).toInt
  }

  def convertPPsToInt(probArray: Array[Double]): Array[Int] = probArray.map{d => convertPPsToInt(d)}

  def readGenLine(line: String, nSamples: Int): (Variant, Annotation, Iterable[Genotype]) = {
    val arr = line.split("\\s+")
    val variant = Variant(arr(0), arr(3).toInt, arr(4), arr(5))
    val annotations = Annotation(arr(2), arr(1)) //rsid, varid
    val dosages = arr.drop(6).map{_.toDouble}

    if (dosages.length != (3 * nSamples))
      fatal("Number of dosages does not match number of samples")

    var plAA = -1
    var plAB = -1
    var plBB = -1
    val plArray = new Array[Int](3)
      val b = new GenotypeStreamBuilder(variant)
      val genoBuilder = new GenotypeBuilder(variant)

      for (i <- dosages.indices by 3) {
        val ints = convertPPsToInt(Array(dosages(i), dosages(i + 1), dosages(i + 2)))

          val pAA = (dosages(i) * 32768).round.toInt
          val pAB = (dosages(i + 1) * 32768).round.toInt
          val pBB = (dosages(i + 2) * 32768).round.toInt

          if (pAA == 32768) {
            plAA = 0
            plAB = 51
            plBB = 51
          } else if (pAB == 32768) {
            plAA = 51
            plAB = 0
            plBB = 51
          } else if (pBB == 32768) {
            plAA = 51
            plAB = 51
            plBB = 0
          } else {
            val dAA = if (pAA == 0) 51 else BgenLoader.phredConversionTable(pAA)
            val dAB = if (pAB == 0) 51 else BgenLoader.phredConversionTable(pAB)
            val dBB = if (pBB == 0) 51 else BgenLoader.phredConversionTable(pBB)

            val minValue = math.min(math.min(dAA, dAB), dBB)

            plAA = (dAA - minValue + .5).toInt
            plAB = (dAB - minValue + .5).toInt
            plBB = (dBB - minValue + .5).toInt
          }

          assert(plAA == 0 || plAB == 0 || plBB == 0)

          val gt = if (plAA == 0 && plAB == 0
            || plAA == 0 && plBB == 0
            || plAB == 0 && plBB == 0)
            -1
          else {
            if (plAA == 0)
              0
            else if (plAB == 0)
              1
            else
              2
          }

        genoBuilder.clear()
        if (gt >= 0) {
          genoBuilder.setGT(gt)
          plArray(0) = plAA
          plArray(1) = plAB
          plArray(2) = plBB
          genoBuilder.setPL(plArray)
        }
        b.write(genoBuilder)

      }
    (variant, annotations, b.result())
  }
}


object GenWriter {

  def apply(outputRoot: String, vds: VariantDataset, sc: SparkContext) {
    val outGenFile = outputRoot + ".gen"
    val outSampleFile = outputRoot + ".sample"
    val hConf = sc.hadoopConfiguration

    writeSampleFile(outSampleFile, hConf, vds.sampleIds)
    writeGenFile(outGenFile, vds)
  }

  def convertPlsToDosage(pl: Array[Int]): Array[Double] = {
    val tmp = pl.map{case p => math.pow(10,p / -10.0)}
    tmp.map{d => d / tmp.sum}
  }

  def appendRow(sb: StringBuilder, v: Variant, va: Annotation, gs: Iterable[Genotype]) {
    sb.append(v.contig)
    sb += ' '
    sb.append("fakeVariantID")
    sb += ' '
    sb.append("fakeRSID")
    sb += ' '
    sb.append(v.start)
    sb += ' '
    sb.append(v.ref)
    sb += ' '
    sb.append(v.alt)

    for (gt <- gs) {
      val dosages = gt.pl match {
        case Some(x) => convertPlsToDosage(x)
        case None => Array(0.0,0.0,0.0)
      }
      sb += ' '
      sb.append(dosages.mkString(" "))
    }
  }

  def writeGenFile(outFile: String, vds: VariantDataset) {
    val kvRDD = vds.rdd.map { case (v, a, gs) =>
      (v, (a, gs.toGenotypeStream(v, compress = false)))
    }
    kvRDD.persist(StorageLevel.MEMORY_AND_DISK)
    kvRDD
      .repartitionAndSortWithinPartitions(new RangePartitioner[Variant, (Annotation, Iterable[Genotype])](vds.rdd.partitions.length, kvRDD))
      .mapPartitions { it: Iterator[(Variant, (Annotation, Iterable[Genotype]))] =>
        val sb = new StringBuilder
        it.map { case (v, (va, gs)) =>
          sb.clear()
          appendRow(sb, v, va, gs)
          sb.result()
        }
      }.writeTable(outFile, None, deleteTmpFiles = true)
    kvRDD.unpersist()
  }

  def writeSampleFile(outFile: String, hConf: Configuration, sampleIds: IndexedSeq[String]){
    val header = Array("ID_1 ID_2 missing","0 0 0")
    writeTable(outFile, hConf, header ++ sampleIds.map{case s => Array(s, s, "0").mkString(" ")})
  }
}
