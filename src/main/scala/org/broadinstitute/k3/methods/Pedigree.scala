package org.broadinstitute.k3.methods

import java.io.{File, FileWriter}

import scala.io.Source

object TryOut {

  def main(args: Array[String]) {
    val ped = Pedigree.read("/Users/Jon/sample_mendel.fam")
    println(ped.trioMap.get("Son1"))
    println(ped.trioMap.get("Mom1"))
    println(ped.trios.trioMap)
    ped.writeSummary("/Users/Jon/sample_mendel.sumfam")
    ped.trios.write("/Users/Jon/sample_mendel_trios.sumfam")
  }
}

object Sex extends Enumeration {
  type Sex = Value
  val Male, Female = Value
}

object Phenotype extends Enumeration {
  type Phenotype = Value
  val Case, Control = Value
}

import org.broadinstitute.k3.methods.Phenotype._
import org.broadinstitute.k3.methods.Sex._

case class TrioData(famID: Option[String], kidID: String, dadID: Option[String], momID: Option[String],
                    sex: Option[Sex], pheno: Option[Phenotype]) {

  def write(fw: FileWriter) = {
    val sexStr: String =
      if (sex == Some(Male)) "1"
      else if (sex == Some(Female)) "2"
      else "0"

    val phenoStr: String =
      if (pheno == Some(Control)) "1"
      else if (pheno == Some(Case)) "2"
      else "0"

    fw.write(famID.getOrElse("0") + " " + kidID + " " + dadID.getOrElse("0") + " " + momID.getOrElse("0") + " " +
        sexStr + " " + phenoStr + "\n")
  }
}

object Pedigree {

  def read(file: String): Pedigree = {
    require(file.endsWith(".fam"))

    new Pedigree(Source.fromFile(new File(file))
      .getLines()
      .filter(line => !line.isEmpty)
      .map{ line =>
      val fields: Array[String] = line.split(" ")
      assert(fields.length == 6)
      val famID = if (fields(0) != "0") Some(fields(0)) else None
      val kidID = fields(1)
      val dadID = if (fields(2) != "0") Some(fields(2)) else None
      val momID = if (fields(3) != "0") Some(fields(3)) else None
      val sex =
        if (fields(4) == "1") Some(Male)
        else if (fields(4) == "2") Some(Female)
        else None
      val pheno = // FIXME: assuming binary phenotype
        if (fields(5) == "1") Some(Control)
        else if (fields(5) == "2") Some(Case)
        else None

      (kidID, TrioData(famID, kidID, dadID, momID, sex, pheno))}
      .toMap)
  }
}

class Pedigree(val trioMap: Map[String, TrioData]) {

  def isMale(kidID: String): Boolean = trioMap(kidID).sex == Some(Male)
  def isFemale(kidID: String): Boolean = trioMap(kidID).sex == Some(Female)
  def noSex(kidID: String): Boolean = trioMap(kidID).sex.isEmpty
  def isCase(kidID: String): Boolean = trioMap(kidID).pheno == Some(Case)
  def isControl(kidID: String): Boolean = trioMap(kidID).pheno == Some(Control)
  def noPheno(kidID: String): Boolean = trioMap(kidID).pheno.isEmpty
  def isTrio(kidID: String): Boolean = trioMap(kidID).dadID.isDefined && trioMap(kidID).momID.isDefined

  def trios: Pedigree = new Pedigree(trioMap.filterKeys(isTrio))

  def nSat(filters: (String => Boolean)*): Int = trioMap.count{ case (k,v) => filters.forall(_(k))}

  // FIXME: nFam based on famID, but can do some inference even when famID's are missing
  def nFam: Int = trioMap.map{ case (k,v) => v.famID }.filter(_.isDefined).toList.distinct.size
  def nIndiv: Int = nSat()
  def nTrio: Int = nSat(isTrio)

  def writeSummary(file: String) = {
    val fw = new FileWriter(new File(file))

    val fields = List("nFam", "nIndiv", "nMale", "nFemale",
      "nTrio", "nMaleTrio", "nFemaleTrio",
      "nCaseTrio", "nControlTrio",
      "nCaseMaleTrio", "nCaseFemaleTrio",
      "nControlMaleTrio", "nControlFemaleTrio")

    val values = List(nFam, nIndiv, nSat(isMale), nSat(isFemale),
      nTrio, nSat(isTrio, isMale), nSat(isTrio, isFemale),
      nSat(isTrio, isCase), nSat(isTrio, isControl),
      nSat(isTrio, isCase, isMale), nSat(isTrio, isCase, isFemale),
      nSat(isTrio, isControl, isMale), nSat(isTrio, isControl, isFemale))

    fw.write(fields.mkString("\t") + "\n")
    fw.write(values.mkString("\t") + "\n")
    fw.close() // FIXME
  }

  def write(file: String) {
    val fw = new FileWriter(new File(file))
    trioMap.values.foreach(_.write(fw))
    fw.close()  // FIXME
  }
}
