package org.broadinstitute.k3.methods

import java.io.FileWriter

import org.apache.spark.rdd.RDD
import org.broadinstitute.k3.variant._

case class MendelError(kidGT: String, dadGT: String, momGT: String) {
  override def toString: String = dadGT + " x " + momGT + " -> " + kidGT
}

case class mMendel(famID: Option[String], kidID: String, chrom: String, varID: String, code: Int, mendelError: MendelError) {
  def write(fw: FileWriter) = {
    fw.write(famID.getOrElse("0") + "\t" + kidID + "\t" + chrom + "\t" + varID + "\t" + code + mendelError.toString + "\n")
  }
}

case class fMendel(famID: Option[String], dadID: String, momID: String, nKid: Int, nError: Int) {
  def write(fw: FileWriter) = {
    fw.write(famID.getOrElse("0") + "\t" + dadID + "\t" + momID + "\t" + nKid + "\t" + nError + "\n")
  }
}

case class iMendel(famID: Option[String], indivID: String, nError: Int) {
  def write(fw: FileWriter) = {
    fw.write(famID.getOrElse("0") + "\t" + indivID + "\t" + nError + "\n")
  }
}

case class lMendel(chrom: String, varID: String, nError: Int) {
  def write(fw: FileWriter) = {
    fw.write(chrom + "\t" + varID + "\t" + nError + "\n")
  }
}

class Mendel(vds: VariantDataset, ped: Pedigree) {

  def roleInTrio(id: String, trio: Trio): Int =
    if (Some(id) == trio.dadID) 1 else 2

  def mendel: RDD[((Trio, Variant), (Map[Int, Genotype], Int))] =
    vds
      .flatMapWithKeys(
        (v,s,g) => {
          val id = vds.sampleIds(s)
          val trio = ped.trioMap(id)
          val triosAsKid = if (trio.hasDadMom) List((trio, 0)) else List()
          val triosAsParent = ped.kidsOfParent(id).map(ped.trioMap(_)).filter(_.hasDadMom).map( t => (t, roleInTrio(id, t)) )

          (triosAsKid ++ triosAsParent).map{ case (t, role) => ((t, v), (role, g))
          }
        })
      .groupByKey()
      .mapValues(_.toMap)
      .map{ case ((t, v), m) => ((t, v), (m, getCode(m(0), m(1), m(2), v.onX))) }

  def getCode(kidGT: Genotype, dadGT: Genotype, momGT: Genotype, onX: Boolean): Int = {
    if (kidGT.isHomRef)
      if (onX && momGT.isHomVar)
        9
      else if (!dadGT.isHomVar && !momGT.isHomVar)
        0
      else if (dadGT.isHomVar && !momGT.isHomVar)
        6
      else if (!dadGT.isHomVar && momGT.isHomVar)
        7
      else
        8
    else if (kidGT.isHet)
      if (dadGT.isHet || momGT.isHet)
        0
      else if (dadGT.isHomRef && momGT.isHomRef)
        2
      else if (dadGT.isHomVar && momGT.isHomVar)
        1
      else
        0
    else if (kidGT.isHomVar)
      if (onX && momGT.isHomRef)
        10
      else if (!dadGT.isHomRef && !momGT.isHomRef)
        0
      else if (dadGT.isHomRef && !momGT.isHomRef)
        3
      else if (!dadGT.isHomRef && momGT.isHomRef)
        4
      else
        5
    else
      0
  }

}
