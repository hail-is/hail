package org.broadinstitute.k3.methods

import java.io.FileWriter

import org.broadinstitute.k3.variant._

case class MendelError(dadGeno: String, momGeno: String, kidGeno: String) {
  override def toString: String = {
    dadGeno + "/" + momGeno + "->" + kidGeno
  }
}

case class mMendel(famID: Option[String], kidID: String, chrom: String, varID: String, code: Int, error: MendelError) {
  def write(fw: FileWriter) = {
    fw.write(famID.getOrElse("0") + " " + kidID + " " + chrom + " " + varID + " " + code +
      error.toString + "\n")
  }
}

case class fMendel()

case class iMendel()

case class lMendel()


object Mendel {
  def mendel(vds: VariantDataset, ped: Pedigree):  = {
    7
  }
}
