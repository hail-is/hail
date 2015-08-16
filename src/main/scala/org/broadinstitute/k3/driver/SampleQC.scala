package org.broadinstitute.k3.driver

import java.io.{File, FileWriter}

import org.broadinstitute.k3.methods._
import org.broadinstitute.k3.variant._

object SampleQC {
  def apply(filename: String, vds: VariantDataset,
            sampleMethods: Array[SampleMethod[Any]]): Unit = {

    val sampleResults = sampleMethods.map(_.apply(vds))

    val fw = new FileWriter(new File(filename))
    fw.write("sampleID" + "\t" + sampleMethods.map(_.name).mkString("\t") + "\n")
    for {
      sampleIndex <- 0 until vds.nSamples
    } yield {
      fw.write(vds.sampleIds(sampleIndex))
      for {
        methodIndex <- 0 until sampleMethods.length
      } yield {fw.write("\t" + sampleResults(methodIndex)(sampleIndex).toString)
      }
      fw.write("\n")
    }
    fw.close()
  }
}
