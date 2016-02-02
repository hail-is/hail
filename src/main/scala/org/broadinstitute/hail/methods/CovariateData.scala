package org.broadinstitute.hail.methods

import org.apache.hadoop
import breeze.linalg._
import org.broadinstitute.hail.Utils._
import scala.io.Source

case class CovariateData(covRowSample: Array[Int], covName: Array[String], data: DenseMatrix[Double]) {

  //preserves increasing order of samples
  def filterSamples(samplesToKeep: Set[Int]): CovariateData = {
    val covRowKeep: Array[Boolean] = covRowSample.map(samplesToKeep)
    val nKeep = covRowKeep.count(identity)
    val nRow = covRowSample.size
    val nSamplesDiscarded = nRow - nKeep

    if (nSamplesDiscarded == 0)
      this
    else {
      val filtCovRowSample = Array.ofDim[Int](nKeep)
      val filtData = DenseMatrix.zeros[Double](nKeep, covName.size)
      var filtRow = 0

      for (row <- 0 until nRow)
        if (covRowKeep(row)) {
          filtCovRowSample(filtRow) = covRowSample(row)
          filtData(filtRow to filtRow, ::) := data(row, ::).t
          filtRow += 1
        }

      warning(s"$nSamplesDiscarded ${plural(nSamplesDiscarded, "sample")} in .cov discarded: missing phenotype.")

      CovariateData(filtCovRowSample, covName, filtData)
    }
  }
}

object CovariateData {

  def read(filename: String, hConf: hadoop.conf.Configuration, sampleIds: IndexedSeq[String]): CovariateData = {
    readFile(filename, hConf) { s =>
      val lines = Source.fromInputStream(s)
        .getLines()
        .filterNot(_.isEmpty)

      val header = lines.next()
      val covName = header.split("\\s+").tail

      var nSamplesDiscarded = 0

      val sampleIndex = sampleIds.zipWithIndex.toMap

      val sampleCovs = collection.mutable.Map[Int, Iterator[Double]]()

      for (line <- lines) {
        val entries = line.split("\\s+").iterator
        val sample = entries.next()
        sampleIndex.get(sample) match {
          case Some(s) =>
            if (sampleCovs.keySet(s))
              fatal(s".cov sample name is not unique: $sample")
            else
              sampleCovs += s -> entries.map(_.toDouble)
          case None => nSamplesDiscarded += 1
        }
      }

      if (nSamplesDiscarded > 0)
        warning(s"$nSamplesDiscarded ${plural(nSamplesDiscarded, "sample")} in .cov discarded: missing from variant data set.")

      val covRowSample = sampleCovs.keys.toArray
      scala.util.Sorting.quickSort(covRowSample) // sorts in place, order preserved by filterSamples

      val data = new DenseMatrix[Double](rows = covRowSample.size, cols = covName.size)
      for (row <- covRowSample.indices)
        for (col <- covName.indices)
          data(row, col) = sampleCovs(covRowSample(row)).next()

      CovariateData(covRowSample, covName, data)
    }
  }
}