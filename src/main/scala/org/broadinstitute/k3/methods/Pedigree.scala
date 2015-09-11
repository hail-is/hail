package org.broadinstitute.k3.methods

import java.io.{File, FileWriter}

import org.broadinstitute.k3.Utils._

import scala.io.Source

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

case class Trio(kidID: String, famID: Option[String], dadID: Option[String], momID: Option[String],
                sex: Option[Sex], pheno: Option[Phenotype]) {

  def isMale: Boolean = sex.contains(Male)
  def isFemale: Boolean = sex.contains(Female)
  def noSex: Boolean = sex.isEmpty
  def isCase: Boolean = pheno.contains(Case)
  def isControl: Boolean = pheno.contains(Control)
  def noPheno: Boolean = pheno.isEmpty
  def hasDad: Boolean = dadID.isDefined
  def hasMom: Boolean = momID.isDefined
  def hasDadMom: Boolean = hasDad && hasMom
}

object Pedigree {
  def apply(trios: Traversable[Trio]): Pedigree = {
    new Pedigree(trios.map(t => t.kidID -> t).toMap)
  }

  def read(file: String): Pedigree = {
    require(file.endsWith(".fam"))
    def maybeField(s: String): Option[String] = if (s != "0") Some(s) else None

    Pedigree(Source.fromFile(new File(file))
      .getLines()
      .filter(line => !line.isEmpty)
      .map { line => // FIXME: proper input error handling (and possible conflicting trio handing)
      val Array(famID, kidID, dadID, momID, sex, pheno) = line.split("\\s+")

      Trio(kidID, maybeField(famID), maybeField(dadID), maybeField(momID),
        Sex.maybeWithName(sex), Phenotype.maybeWithName(pheno))
      }
      .toTraversable
    )
  }
}

case class Pedigree(trioMap: Map[String, Trio]) {

  def kidsOfParent: Map[String, List[String]] =
    trios.flatMap(t => t.momID.map(_ -> t.kidID).toList ++ t.dadID.map(_ -> t.kidID).toList)
      .groupBy(_._1)   // FIXME: implement groupByKey
      .map{ case (parentID, parentKidIDs) => (parentID, parentKidIDs.map(_._2).toList) }
      .withDefaultValue(List())

  def trios = trioMap.values

  def dadOf(sampleIds: Array[String]): Map[Int, Int] = {
    val sampleIndices = sampleIds.zipWithIndex.toMap
    trios
      .flatMap{ t => t.dadID.map{ id => (sampleIndices(t.kidID), sampleIndices(id)) } }
      .toMap
  }

  def momOf(sampleIds: Array[String]): Map[Int, Int] = {
    val sampleIndices = sampleIds.zipWithIndex.toMap
    trios
      .flatMap{ t => t.momID.map{ id => (sampleIndices(t.kidID), sampleIndices(id)) } }
      .toMap
  }

  def nSatisfying(filters: (Trio => Boolean)*): Int = trioMap.count{ case (k,v) => filters.forall(_(v)) }
  def nFam: Int = trioMap.flatMap{ case (k,v) => v.famID }.toSet.size  // FIXME: add distinct
  def nIndiv: Int = trioMap.size
  def nCompleteTrio: Int = nSatisfying(_.hasDadMom)

  def writeSummary(filename: String) = {
    val fw = new FileWriter(new File(filename))

    val columns = List(
      ("nFam", nFam), ("nIndiv", nIndiv), ("nCompleteTrios", nCompleteTrio),
      ("nMale", nSatisfying(_.isMale)), ("nFemale", nSatisfying(_.isFemale)),
      ("nCase", nSatisfying(_.isCase)), ("nControl", nSatisfying(_.isControl)),
      ("nMaleTrio", nSatisfying(_.hasDadMom, _.isMale)),
      ("nFemaleTrio", nSatisfying(_.hasDadMom, _.isFemale)),
      ("nCaseTrio", nSatisfying(_.hasDadMom, _.isCase)),
      ("nControlTrio", nSatisfying(_.hasDadMom, _.isControl)),
      ("nCaseMaleTrio", nSatisfying(_.hasDadMom, _.isCase, _.isMale)),
      ("nCaseFemaleTrio", nSatisfying(_.hasDadMom, _.isCase, _.isFemale)),
      ("nControlMaleTrio", nSatisfying(_.hasDadMom, _.isControl, _.isMale)),
      ("nControlFemaleTrio", nSatisfying(_.hasDadMom, _.isControl, _.isFemale)))

    withFileWriter(filename){ fw =>
      fw.write(columns.map(_._1).mkString("\t") + "\n")
      fw.write(columns.map(_._2).mkString("\t") + "\n")
    }
  }

  // FIXME: no header in plink fam file, but "FID\tKID\tPAT\tMAT\tSEX\tPHENO" sure seems appropriate
  def write(filename: String) {
    def trioLine(t: Trio): String =
      t.famID.getOrElse("0") + "\t" + t.kidID + "\t" + t.dadID.getOrElse("0") + "\t" +
        t.momID.getOrElse("0") + "\t" + t.sex.getOrElse("0") + "\t" + t.pheno.getOrElse("0") + "\n"
    
    val lines = trioMap.values.map(trioLine)
    writeTable(filename, lines)
  }
}
