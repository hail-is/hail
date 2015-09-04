package org.broadinstitute.k3.methods

import java.io.{File, FileWriter}

import org.broadinstitute.k3.Utils._

import scala.io.Source

object TryOutPedigree {

  def main(args: Array[String]) {
    val ped = Pedigree.read("src/test/resources/sample_mendel.fam")
    ped.writeSummary("/tmp/sample_mendal.sumfam")
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

case class Trio(famID: Option[String], kidID: String, dadID: Option[String], momID: Option[String],
                sex: Option[Sex], pheno: Option[Phenotype]) {

  def write(fw: FileWriter) =
    fw.write(famID.getOrElse("0") + "\t" + kidID + "\t" + dadID.getOrElse("0") + "\t" + momID.getOrElse("0") + "\t" +
      sex.getOrElse("0") + "\t" + pheno.getOrElse("0") + "\n")

  //
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

      Trio(maybeField(famID), kidID, maybeField(dadID), maybeField(momID),
        Sex.maybeWithName(sex), Phenotype.maybeWithName(pheno))
      }
      .toTraversable
    )
  }
}

case class Pedigree(trioMap: Map[String, Trio]) {

  override def equals(that: Any): Boolean = that match {
    case that: Pedigree => this.trioMap == that.trioMap
    case _ => false
  }

  val kidsOfParent: Map[String, List[String]] =
    trios.flatMap(t => t.momID.map(_ -> t.kidID).toList ++ t.dadID.map(_ -> t.kidID).toList)
      .groupBy(_._1)   // FIXME: implement groupByKey
      .map{ case (parentID, parentKidIDs) => (parentID, parentKidIDs.map(_._2).toList) }
      .withDefaultValue(List())

  override def toString = trioMap.toString()

  def trios = trioMap.values

  def nSatisfying(filters: (Trio => Boolean)*): Int = trioMap.count{ case (k,v) => filters.forall(_(v)) }

  // FIXME: nFam based on famID, but can do some inference even when famID's are missing
  def nFam: Int = trioMap.map{ case (k,v) => v.famID }.filter(_.isDefined).toSet.size  // FIXME: add distinct
  def nIndiv: Int = trioMap.size
  def nBothParents: Int = nSatisfying(_.hasDadMom)

  def writeSummary(file: String) = {
    val fw = new FileWriter(new File(file))

    val columns = List(
      ("nFam", nFam), ("nIndiv", nIndiv), ("nTrio", nBothParents),
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

    fw.write(columns.map(_._1).mkString("\t") + "\n")
    fw.write(columns.map(_._2).mkString("\t") + "\n")
    fw.close() // FIXME
  }

  def write(filename: String) {
    cleanly[FileWriter, Unit](new FileWriter(new File(filename)),
    fw => fw.close(), { fw =>
      trioMap.values.foreach(_.write(fw))
    })
  }
}
