package org.broadinstitute.hail.methods

import org.apache.hadoop
import breeze.linalg._
import org.broadinstitute.hail.RichDenseMatrixDouble
import org.broadinstitute.hail.Utils._
import scala.io.Source

case class CovariateData(covRowSample: Array[String], covName: Array[String], data: Option[DenseMatrix[Double]]) {
  require(data.isDefined || covRowSample.isEmpty || covName.isEmpty)
  require(data.forall(m => m.rows == covRowSample.size && m.cols == covName.size))
  require(covRowSample.areDistinct())
  require(covName.areDistinct())

  //preserves increasing order of samples
  def filterSamples(keepSample: String => Boolean): CovariateData = {
    val filtCovRowSample = covRowSample.filter(keepSample)
    val nSamplesDiscarded = covRowSample.size - filtCovRowSample.size

    if (nSamplesDiscarded == 0)
      this
    else {
      warn(s"$nSamplesDiscarded ${plural(nSamplesDiscarded, "sample")} in .cov discarded: missing phenotype.")

      CovariateData(filtCovRowSample, covName, data.flatMap(_.filterRows(row => keepSample(covRowSample(row)))))
    }
  }

  def filterCovariates(keepCov: String => Boolean): CovariateData =
    CovariateData(covRowSample, covName.filter(keepCov), data.flatMap(_.filterCols(col => keepCov(covName(col)))))

  def appendCovariates(that: CovariateData): CovariateData = {
    fatalIf(!(this.covRowSample sameElements that.covRowSample),
      "Cannot append covariates: samples (rows) are not aligned.")

    val newCovName = this.covName ++ that.covName

    fatalIf(!newCovName.areDistinct(),
      s"Cannot append covariates: covariate names overlap for ${newCovName.duplicates()}")

    CovariateData(covRowSample, newCovName, RichDenseMatrixDouble.horzcat(this.data, that.data))
  }
}

object CovariateData {

  def read(filename: String, hConf: hadoop.conf.Configuration, sampleIds: IndexedSeq[String]): CovariateData = {
    readFile(filename, hConf) { input =>
      val lines = Source.fromInputStream(input)
        .getLines()
        .filterNot(_.isEmpty)

      val header = lines.next()
      val covName = header.split("\t").tail

      if (!covName.areDistinct()) {
        fatal(s"Error on covariate import: covariate names occur multiple times for ${covName.duplicates()}")
      }

      var nSamplesDiscarded = 0

      val sampleSet = sampleIds.toSet

      val sampleCovs = collection.mutable.Map[String, Iterator[Double]]()

      for (line <- lines) {
        val entries = line.split("\t").iterator
        val sample = entries.next()
        sampleSet(sample) match {
          case true =>
            if (sampleCovs.contains(sample))
              fatal(s".cov sample name is not unique: $sample")
            else
              sampleCovs += sample -> entries.map(_.toDouble)
          case false => nSamplesDiscarded += 1
        }
      }

      if (nSamplesDiscarded > 0)
        warn(s"$nSamplesDiscarded ${plural(nSamplesDiscarded, "sample")} in .cov discarded: missing from variant data set.")

      val covRowSample = sampleCovs.keys.toArray
      scala.util.Sorting.quickSort(covRowSample) // sorts in place, order preserved by filterSamples

      val data =
        if (covRowSample.isEmpty || covName.isEmpty)
          None
        else
          Some {
            val d = new DenseMatrix[Double](rows = covRowSample.size, cols = covName.size)
            for (row <- covRowSample.indices)
              for (col <- covName.indices)
                d(row, col) = sampleCovs(covRowSample(row)).next()
            d
          }

      CovariateData(covRowSample, covName, data)
    }
  }
}