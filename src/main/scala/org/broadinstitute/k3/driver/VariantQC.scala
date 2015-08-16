package org.broadinstitute.k3.driver

import java.io.{File, FileWriter}

import org.broadinstitute.k3.methods._
import org.broadinstitute.k3.variant._

object VariantQC {
  def apply(filename: String, vds: VariantDataset,
            variantMethods: Array[VariantMethod[Any]]): Unit = {

    val variantResults = variantMethods.map(_.apply(vds))

    val header = "Chrom" + "\t" + "Pos" + "\t" + "Ref" + "\t" + "Alt" + "\t" + variantMethods.map(_.name).mkString("\t") + "\n"

    val fw = new FileWriter(new File(filename))
    fw.write(header)
    for {
      variant <- vds.variants.sortBy(v => (v.contig, v.start, v.ref, v.alt))
    } yield {
      fw.write(variant.contig + "\t" + variant.start + "\t" + variant.ref + "\t" + variant.alt)
      for {
        methodIndex <- variantMethods.indices
      } yield {fw.write("\t" + variantResults(methodIndex)(variant).toString)
      }
      fw.write("\n")
    }
    fw.close()
  }
}
