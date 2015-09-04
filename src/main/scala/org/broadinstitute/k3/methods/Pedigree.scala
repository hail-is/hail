package org.broadinstitute.k3.methods

import java.io.{File, FileWriter}

import org.broadinstitute.k3.Utils._

import scala.io.Source

object TryOut {

  def main(args: Array[String]) {
    val ped = Pedigree.read("src/test/resources/sample_mendel.fam")
    ped.write("src/test/resources/sample_mendel2.fam")
    val ped2 = Pedigree.read("src/test/resources/sample_mendel2.fam")
    println(ped)
    println(ped == ped2)
  }
}

object Sex extends Enumeration {
  type Sex = Value
  val Male = Value("1")
  val Female = Value("2")
}

object Phenotype extends Enumeration {
  type Phenotype = Value
  val Control = Value("1")
  val Case = Value("2")
}

import org.broadinstitute.k3.methods.Phenotype._
import org.broadinstitute.k3.methods.Sex._

case class TrioData(famID: Option[String], kidID: String, dadID: Option[String], momID: Option[String],
                    sex: Option[Sex], pheno: Option[Phenotype]) {

  def write(fw: FileWriter) =
    fw.write(famID.getOrElse("0") + "\t" + kidID + "\t" + dadID.getOrElse("0") + "\t" + momID.getOrElse("0") + "\t" +
        sex.getOrElse("0") + "\t" + pheno.getOrElse("0") + "\n")

  def isMale: Boolean = sex == Some(Male)
  def isFemale: Boolean = sex == Some(Female)
  def noSex: Boolean = sex.isEmpty
  def isCase: Boolean = pheno == Some(Case)
  def isControl: Boolean = pheno == Some(Control)
  def noPheno: Boolean = pheno.isEmpty
  def isTrio: Boolean = dadID.isDefined && momID.isDefined
}

object Pedigree {
  def read(file: String): Pedigree = {  // FIXME: check for non-identical lines with same kidID
    require(file.endsWith(".fam"))

    new Pedigree(Source.fromFile(new File(file))
      .getLines()
      .filter(line => !line.isEmpty)
      .map{ line =>
      val fields: Array[String] = line.split("\\s+")
      assert(fields.length == 6)
      val famID = if (fields(0) != "0") Some(fields(0)) else None
      val kidID = fields(1)
      val dadID = if (fields(2) != "0") Some(fields(2)) else None
      val momID = if (fields(3) != "0") Some(fields(3)) else None
      val sex = Sex.maybeWithName(fields(4))
      val pheno = Phenotype.maybeWithName(fields(5))

      (kidID, TrioData(famID, kidID, dadID, momID, sex, pheno))}
      .toMap)
  }
}

class Pedigree(val trioMap: Map[KidID, TrioData]) {

  override def equals(that: Any): Boolean = that match {
    case that: Pedigree => this.trioMap == that.trioMap
    case _ => false
  }

  override def toString = trioMap.toString()

  def trios: Pedigree = new Pedigree(trioMap.filter{ case (k,v) => v.isTrio })

  def nSat(filters: (TrioData => Boolean)*): Int = trioMap.count{ case (k,v) => filters.forall(_(v)) }

  // FIXME: nFam based on famID, but can do some inference even when famID's are missing
  def nFam: Int = trioMap.map{ case (k,v) => v.famID }.filter(_.isDefined).toSet.size  // FIXME: add distinct
  def nIndiv: Int = trioMap.size
  def nTrio: Int = nSat(_.isTrio)

  def writeSummary(file: String) = {
    val fw = new FileWriter(new File(file))

    val columns = List(
      ("nFam", nFam), ("nIndiv", nIndiv), ("nTrio", nTrio),
      ("nMale", nSat(_.isMale)), ("nFemale", nSat(_.isFemale)),
      ("nCase", nSat(_.isCase)), ("nControl", nSat(_.isControl)),
      ("nMaleTrio", nSat(_.isTrio, _.isMale)), ("nFemaleTrio", nSat(_.isTrio, _.isFemale)),
      ("nCaseTrio", nSat(_.isTrio, _.isCase)), ("nControlTrio", nSat(_.isTrio, _.isControl)),
      ("nCaseMaleTrio", nSat(_.isTrio, _.isCase, _.isMale)),
      ("nCaseFemaleTrio", nSat(_.isTrio, _.isCase, _.isFemale)),
      ("nControlMaleTrio", nSat(_.isTrio, _.isControl, _.isMale)),
      ("nControlFemaleTrio", nSat(_.isTrio, _.isControl, _.isFemale)))

    fw.write(columns.map(_._1).mkString("\t") + "\n")
    fw.write(columns.map(_._2).mkString("\t") + "\n")
    fw.close() // FIXME
  }

  def write(file: String) {
    val fw = new FileWriter(new File(file))
    trioMap.values.foreach(_.write(fw))
    fw.close()  // FIXME
  }
}
