package org.broadinstitute.hail.methods

import org.apache.hadoop
import breeze.linalg._
import org.broadinstitute.hail.Utils._
import scala.io.Source

case class CovariateData(covRowSample: Array[Int], covName: Array[String], data: Option[DenseMatrix[Double]]) {
  require(data.isDefined || covRowSample.isEmpty || covName.isEmpty)
  require(data.forall(m => m.rows == covRowSample.size && m.cols == covName.size))
  require(covRowSample.areDistinct())
  require(covName.areDistinct())

  //preserves increasing order of samples
  def filterSamples(samplesToKeep: Set[Int]): CovariateData = {
    val covRowKeep: Array[Boolean] = covRowSample.map(samplesToKeep)
    val nKeep = covRowKeep.count(identity)

    if (covRowSample.size == nKeep)
      this
    else if (nKeep == 0)
      CovariateData(Array[Int](), covName, None)
    else if (data.isEmpty)
      CovariateData(covRowSample.filter(samplesToKeep), covName, None)
    else {
      val filtCovRowSample = Array.ofDim[Int](nKeep)
      val filtData = DenseMatrix.zeros[Double](nKeep, covName.size)
      var filtRow = 0

      for (row <- covRowSample.indices)
        if (covRowKeep(row)) {
          filtCovRowSample(filtRow) = covRowSample(row)
          filtData(filtRow to filtRow, ::) := data.get(row, ::).t
          filtRow += 1
        }

      val nSamplesDiscarded = covRowSample.size - nKeep
      // FIXME: in the future there may be other reasons
      warn(s"$nSamplesDiscarded ${plural(nSamplesDiscarded, "sample")} in .cov discarded: missing phenotype.")

      CovariateData(filtCovRowSample, covName, Some(filtData))
    }
  }

  def filterCovariates(covsToKeep: Set[String]): CovariateData = {
    val covColKeep: Array[Boolean] = covName.map(covsToKeep)
    val nKeep = covColKeep.count(identity)

    if (covName.size == nKeep)
      this
    else if (nKeep == 0)
      CovariateData(covRowSample, Array[String](), None)
    else if (data.isEmpty)
      CovariateData(covRowSample, covName.filter(covsToKeep), None)
    else {
      val filtCovName = Array.ofDim[String](nKeep)
      val filtData = DenseMatrix.zeros[Double](covRowSample.size, nKeep)
      var filtCol = 0

      for (col <- covName.indices)
        if (covColKeep(col)) {
          filtCovName(filtCol) = covName(col)
          filtData(::, filtCol to filtCol) := data.get(::, col)
          filtCol += 1
        }

      CovariateData(covRowSample, filtCovName, Some(filtData))
    }
  }

  def appendCovariates(that: CovariateData): CovariateData = {
    if (!(this.covRowSample sameElements that.covRowSample))
      fatal("Cannot append covariates: samples (rows) are not aligned.")

    val newCovName = this.covName ++ that.covName

    if (!newCovName.areDistinct())
      fatal(s"Cannot append covariates: covariate names overlap for ${newCovName.duplicates()}")

    val newData = (this.data, that.data) match {
      case (Some(d1), Some(d2)) => Some(DenseMatrix.horzcat(d1,d2))
      case (Some(d1), None) => this.data
      case (None, Some(d2)) => that.data
      case (None, None) => None
    }

    CovariateData(covRowSample, newCovName, newData)
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

      val sampleIndex = sampleIds.zipWithIndex.toMap

      val sampleCovs = collection.mutable.Map[Int, Iterator[Double]]()

      for (line <- lines) {
        val entries = line.split("\t").iterator
        val sample = entries.next()
        sampleIndex.get(sample) match {
          case Some(s) =>
            if (sampleCovs.contains(s))
              fatal(s".cov sample name is not unique: $sample")
            else
              sampleCovs += s -> entries.map(_.toDouble)
          case None => nSamplesDiscarded += 1
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