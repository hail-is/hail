package org.broadinstitute.k3.driver

import java.io.{File, FileWriter}

import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.methods._
import org.broadinstitute.k3.variant._

object VariantQC {
  def apply(filename: String, vds: VariantDataset,
            results: (Array[String], RDD[(Variant, Array[Any])])) {

    val fw = new FileWriter(new File(filename + ".header"))

    val header = "Chrom" + "\t" + "Pos" + "\t" + "Ref" + "\t" + "Alt" + "\t" + results._1.mkString("\t") + "\n"
    fw.write(header)

    fw.close()

    results._2
    .map{ case (v, a) => (Array[Any](v.contig, v.start, v.ref, v.alt) ++ a).mkString("\t") }
    .saveAsTextFile(filename)
  }
}
