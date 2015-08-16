package org.broadinstitute.k3.driver

import java.io.{File, FileWriter}

import org.apache.spark.SparkContext
import org.broadinstitute.k3.methods._
import org.broadinstitute.k3.variant._

object SampleQC {
  def apply(filename: String, vds: VariantDataset,
            sampleMethods: Array[SampleMethod[Any]]): Unit = {
    val sampleResults: Array[Map[Int, Any]] = sampleMethods.map(_.apply(vds))

    val fw = new FileWriter(new File(filename))

    val header = "sampleID" + "\t" + sampleMethods.map(_.name).mkString("\t") + "\n"
    fw.write(header)

    for ((id, i) <- vds.sampleIds.zipWithIndex) {
      fw.write(id)
      for (r <- sampleResults)
        fw.write("\t" + r(i).toString)
      fw.write("\n")
    }

    // FIXME
    fw.close()
  }
}
