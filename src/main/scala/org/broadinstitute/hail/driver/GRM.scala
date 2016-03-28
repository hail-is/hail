package org.broadinstitute.hail.driver

import breeze.linalg.DenseMatrix
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.methods.ToStandardizedIndexedRowMatrix
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object GRM extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _
  }

  def newOptions = new Options

  def name = "grm"

  def description = "Compute the Genetic Relatedness Matrix (GRM)"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val (variants, mat) = ToStandardizedIndexedRowMatrix(vds)
    
    val n = vds.nSamples
    val grm = mat.rows.mapPartitions { it =>
      val b = new mutable.ArrayBuilder.ofDouble
      var i = 0
      it.foreach { r =>
        i += 1
        b ++= r.vector.toArray
      }
      if (i == 0)
        Iterator(DenseMatrix.zeros[Double](n, n))
      else {
        val m = new DenseMatrix(n, i, b.result())
        Iterator(m * m.t)
      }
    }.treeAggregate[DenseMatrix[Double]](DenseMatrix.zeros[Double](n, n))(
      { case (m1, m2) => m1 + m2 },
      { case (m1, m2) => m1 + m2 })

    writeTextFile(options.output, state.hadoopConf) { s =>
      for (i <- 0 until n) {
        for (j <- 0 until n) {
          if (j > 0)
            s.write(" ")
          s.write(grm(i, j).toString)
        }
        s.write("\n")
      }
    }

    state
  }
}
