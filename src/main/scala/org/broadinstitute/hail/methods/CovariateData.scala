package org.broadinstitute.hail.methods

import org.apache.hadoop
import breeze.linalg._
import org.broadinstitute.hail.Utils._

import scala.collection.mutable.ArrayBuffer
import scala.io.Source

case class CovariateData(covRowSample: Array[Int], covName: Array[String], data: DenseMatrix[Double]) {

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

      warning((if (nSamplesDiscarded > 1) s"$nSamplesDiscarded samples" else "1 sample") +
        " in .cov discarded: missing phenotype.")

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

      val covRowSampleBuffer = new ArrayBuffer[Int]
      val dataBuffer = new ArrayBuffer[Double]
      var nSamplesDiscarded = 0

      val sampleIndex = sampleIds.zipWithIndex.toMap
      val sampleSet = collection.mutable.Set[Int]()

      for (line <- lines) {
        val entries = line.split("\\s+").iterator
        val sample = entries.next()
        sampleIndex.get(sample) match {
          case Some(s) =>
            if (sampleSet(s))
              fatal(".cov sample name is not unique: " + sample)
            else
              sampleSet += s
            covRowSampleBuffer += s
            dataBuffer ++= entries.map(_.toDouble)
          case None => nSamplesDiscarded += 1
        }
      }

      val covRowSample = covRowSampleBuffer.toArray

      if (nSamplesDiscarded > 0)
        warning((if (nSamplesDiscarded > 1) s"$nSamplesDiscarded samples" else "1 sample") +
          " in .cov discarded: missing from variant data set.")

      val data = new DenseMatrix[Double](rows = covRowSample.size, cols = covName.size, data = dataBuffer.toArray,
        offset = 0, majorStride = covName.size, isTranspose = true)

      CovariateData(covRowSample, covName, data)
    }
  }
}