package org.broadinstitute.k3.driver

import java.io.{File, FileWriter}

import org.broadinstitute.k3.methods._
import org.broadinstitute.k3.variant._

object VariantQC {
  def apply(filename: String, vds: VariantDataset,
            variantResults: Array[(String, Map[Variant, Any])]) {

    val fw = new FileWriter(new File(filename))

    val header = "Chrom" + "\t" + "Pos" + "\t" + "Ref" + "\t" + "Alt" + "\t" + variantResults.map(_._1).mkString("\t") + "\n"
    fw.write(header)

    for(variant <- vds.variants) {
      fw.write(variant.contig + "\t" + variant.start + "\t" + variant.ref + "\t" + variant.alt)
      for ((m, r) <- variantResults)
        fw.write("\t" + r(variant).toString)
      fw.write("\n")
    }

    fw.close()
  }
}
