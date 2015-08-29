package org.broadinstitute.k3.methods

import java.io.{File, FileWriter}

import scala.io.Source

object TryOut {
  def main(args: Array[String]) {
    val ped = Pedigree.read("/Users/Jon/sample.fam")
    println(ped.pedMap.get("MIGFI_1014"))
    println(ped.nTrio)
    ped.write("/Users/Jon/sample_output.ped")
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

import Sex._
import Phenotype._

case class TrioData(famID: String, dadID: Option[String], momID: Option[String], sex: Option[Sex], phenotype: Option[Phenotype])

object Pedigree {

  def read(file: String): Pedigree = {
    require(file.endsWith(".fam"))

    new Pedigree(Source.fromFile(new File(file))
      .getLines()
      .filter(line => !line.isEmpty)
      .map{ line =>
      val fields: Array[String] = line.split(" ")
      assert(fields.length == 6)
      val famID = fields(0)
      val kidID = fields(1)
      val dadID = if (fields(2) != "0") Some(fields(2)) else None  // '0' if father not in dataset  // FIXME: use Option
      val momID = if (fields(3) != "0") Some(fields(2)) else None // '0' if mother not in dataset
      val sex = if (fields(4) == "1") Some(Male)
        else if (fields(4) == "2") Some(Female)
        else None
      val phenotype = if (fields(4) == "1") Some(Control) //FIXME: assuming binary phenotype
        else if (fields(4) == "2") Some(Case)
        else None

      (kidID, TrioData(famID, dadID, momID, sex, phenotype))}
      .toMap)
  }
}

class Pedigree(val pedMap: Map[String, TrioData]) {

  def getFamID(kidID: String): String = pedMap(kidID).famID
  def getDadID(kidID: String): Option[String] = pedMap(kidID).dadID
  def getMomID(kidID: String): Option[String] = pedMap(kidID).momID
  def getSex(kidID: String): Option[Sex]  = pedMap(kidID).sex
  def getPhenotype(kidID: String): Option[Phenotype] = pedMap(kidID).phenotype

  def isTrio(kidID: String): Boolean =
    getDadID(kidID).isDefined && getMomID(kidID).isDefined

  def nMale: Int = pedMap.filterKeys(getSex(_) == Some(Male)).size
  def nFemale: Int = pedMap.filterKeys(getSex(_) == Some(Female)).size
  def nCase: Int = pedMap.filterKeys(getPhenotype(_) == Some(Case)).size
  def nControl: Int = pedMap.filterKeys(getPhenotype(_) == Some(Control)).size
  def nTrio: Int = pedMap.filterKeys(isTrio).size

  // FIXME: add support for distinct trios, quads, ... ?

  def write(file: String) {
    val fw = new FileWriter(new File(file))

    pedMap
      .foreach { case (kidID, TrioData(famID, dadID, momID, sex, phenotype)) =>
      fw.write(famID + " " + kidID + " " + dadID + " " + momID + " " + sex + " " + phenotype + "\n")
      //fw.write(List(famID, kidID, dadID, momID, sex.toString, phenotype.toString).mkString(" ") + "\n")
    }

    // FIXME
    fw.close()
  }
}
