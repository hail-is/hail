package org.broadinstitute.k3.methods

import java.io.{File, FileWriter}

import org.broadinstitute.k3.Utils._
import org.broadinstitute.k3.methods.Role._

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

case class Trio(kid: Int, fam: Option[String], dad: Option[Int], mom: Option[Int],
                sex: Option[Sex], pheno: Option[Phenotype]) {

  def contains(s: Int): Boolean = kid == s || dad.contains(s) || mom.contains(s)
  def role(s: Int): Role =
    if      (s == kid)        Kid
    else if (dad.contains(s)) Dad
    else                      Mom
  def isMale: Boolean = sex.contains(Male)
  def isFemale: Boolean = sex.contains(Female)
  def noSex: Boolean = sex.isEmpty
  def isCase: Boolean = pheno.contains(Case)
  def isControl: Boolean = pheno.contains(Control)
  def noPheno: Boolean = pheno.isEmpty
  def hasDad: Boolean = dad.isDefined
  def hasMom: Boolean = mom.isDefined
  def hasDadMom: Boolean = hasDad && hasMom
}

object Pedigree {
  def apply(trios: Traversable[Trio]): Pedigree = {
    new Pedigree(trios.map(t => t.kid -> t).toMap)
  }

  def read(filename: String, sampleIds: Array[String]): Pedigree = {
    require(filename.endsWith(".fam"))

    val indexOfSample = sampleIds.zipWithIndex.toMap

    def maybeIndiv(s: String) = if (s != "0") Some(indexOfSample(s)) else None
    def maybeFam(s: String) = if (s != "0") Some(s) else None

    Pedigree(Source.fromFile(new File(filename))
      .getLines()
      .filter(line => !line.isEmpty)
      .map { line => // FIXME: proper input error handling (and possible conflicting trio handing)
      val Array(fam, kid, dad, mom, sex, pheno) = line.split("\\s+")

      Trio(indexOfSample(kid), maybeFam(fam), maybeIndiv(dad), maybeIndiv(mom),
        Sex.maybeWithName(sex), Phenotype.maybeWithName(pheno))
      }
      .toTraversable
    )
  }
}

case class Pedigree(trioMap: Map[Int, Trio]) {

  def trios = trioMap.values

  def completeTriosContaining(s: Int) = trios.filter(t => t.hasDadMom && t.contains(s))

  def dadOf: Map[Int, Int] = trios.flatMap{ t => t.dad.map{ id => (t.kid, id) } }.toMap
  def momOf: Map[Int, Int] = trios.flatMap{ t => t.mom.map{ id => (t.kid, id) } }.toMap

  def nSatisfying(filters: (Trio => Boolean)*): Int = trioMap.count{ case (k,v) => filters.forall(_(v)) }
  def nFam: Int = trioMap.flatMap{ case (k,v) => v.fam }.toSet.size  // FIXME: add distinct
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
  def write(filename: String, sampleIds: Array[String]) {
    def stringIdOrElse(i: Option[Int]) = if (i.isDefined) sampleIds(i.get) else "0"

    def trioLine(t: Trio): String =
      t.fam.getOrElse("0") + "\t" + sampleIds(t.kid) + "\t" + stringIdOrElse(t.dad) + "\t" +
        stringIdOrElse(t.mom) + "\t" + t.sex.getOrElse("0") + "\t" + t.pheno.getOrElse("0") + "\n"

    val lines = trioMap.values.map(trioLine)
    writeTable(filename, lines)
  }
}
