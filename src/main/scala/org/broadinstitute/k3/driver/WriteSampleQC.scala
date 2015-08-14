package org.broadinstitute.k3.driver

import java.io.{File, FileWriter}

import org.broadinstitute.k3.methods._
import org.broadinstitute.k3.variant._

object WriteSampleQC {
  def apply(filename: String, vds: VariantDataset, sampleMethods: Array[SampleMethod]): Unit = {

    val samples = vds.sampleIds
    val sampleResults = sampleMethods.map(_.apply(vds))

    val fw = new FileWriter(new File(filename))
    fw.write(sampleMethods.map(_.name).toString)
    fw.close()
  }
}
