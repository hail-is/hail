package org.broadinstitute.k3.methods

import scala.io.Source
import org.apache.spark.{SparkConf, SparkContext}
import org.broadinstitute.k3.variant._
import org.broadinstitute.k3.Utils._
import org.broadinstitute.k3.vcf

object LoadVCF {
  // FIXME move to VariantDataset
  def apply(sc: SparkContext,
            file: String,
            readerBuilder: vcf.AbstractRecordReaderBuilder = vcf.HtsjdkRecordReaderBuilder,
            vsmtype: String = "sparky",
            compress: Boolean = true,
            nPartitions: Option[Int] = None): VariantDataset = {

    require(file.endsWith(".vcf")
      || file.endsWith(".vcf.bgz")
      || file.endsWith(".vcf.gz"))

    val hConf = sc.hadoopConfiguration
    val headerLines = readFile(file, hConf) { s =>
      Source.fromInputStream(s)
        .getLines()
        .takeWhile(line => line(0) == '#')
        .toArray
    }

    val headerLine = headerLines.last
    assert(headerLine(0) == '#' && headerLine(1) != '#')

    val sampleIds = headerLine
      .split("\t")
      .drop(9)

    val headerLinesBc = sc.broadcast(headerLines)
    val genotypes = sc.textFile(file, nPartitions.getOrElse(sc.defaultMinPartitions))
      .mapPartitions { lines =>
        val reader = readerBuilder.result(headerLinesBc.value)
        lines.filter(line => !line.isEmpty && line(0) != '#')
          .flatMap(reader.readRecord)
          .map { case (v, gs) =>
            val b = new GenotypeStreamBuilder(v, compress)
            for (g <- gs)
              b += 0 -> g
            (v, b.result())
          }
      }

    // FIXME null should be contig lengths
    VariantSampleMatrix(vsmtype, VariantMetadata(null, sampleIds, headerLines), genotypes)
  }
}
