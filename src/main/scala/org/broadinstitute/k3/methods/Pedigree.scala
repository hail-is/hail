package org.broadinstitute.k3.methods

object Role extends Enumeration {
  type Role = Value
  val Kid = Value("0")
  val Dad = Value("1")
  val Mom = Value("2")
}

import java.io.{File, FileWriter}

import org.broadinstitute.k3.Utils._
import org.broadinstitute.k3.methods.Role._
import org.broadinstitute.k3.variant.{Phenotype, Sex}

import scala.io.Source

import Phenotype._
import Sex._

case class Trio(kid: Int, fam: Option[String], dad: Option[Int], mom: Option[Int],
                sex: Option[Sex], pheno: Option[Phenotype]) {

  def contains(s: Int): Boolean =
    kid == s || dad.contains(s) || mom.contains(s)

  def role(s: Int): Option[Role] =
    if      (s == kid)        Some(Kid)
    else if (dad.contains(s)) Some(Dad)
    else if (mom.contains(s)) Some(Mom)
    else                      None

  def isMale: Boolean = sex.contains(Male)
  def isFemale: Boolean = sex.contains(Female)
  def isCase: Boolean = pheno.contains(Case)
  def isControl: Boolean = pheno.contains(Control)
  def isComplete: Boolean = dad.isDefined && mom.isDefined
}

object Pedigree {
  def apply(trios: Traversable[Trio]): Pedigree =
    new Pedigree(trios.map(t => t.kid -> t).toMap)

  def read(filename: String, sampleIds: Array[String]): Pedigree = {
    require(filename.endsWith(".fam"))

    val indexOfSample: Map[String, Int] = sampleIds.zipWithIndex.toMap

    def maybeId(id: String): Option[Int] = if (id != "0") indexOfSample.get(id) else None
    def maybeFam(fam: String): Option[String] = if (fam != "0") Some(fam) else None

    Pedigree(Source.fromFile(new File(filename))
      .getLines()
      .filter(line => !line.isEmpty)
      .map { line => // FIXME: proper input error handling (and possible conflicting trio handing)
      val Array(fam, kid, dad, mom, sex, pheno) = line.split("\\s+")

      Trio(indexOfSample(kid), maybeFam(fam), maybeId(dad), maybeId(mom),
        Sex.withNameOption(sex), Phenotype.withNameOption(pheno))
      }
      .toTraversable
    )
  }
}

case class Pedigree(trioMap: Map[Int, Trio]) {

  def trios = trioMap.values
  def completeTrios = trios.filter(_.isComplete)
  def nuclearFams: Map[(Int, Int), List[Int]] =
    completeTrios
      .map(t => ((t.dad.get, t.mom.get), t.kid))
      .groupBy(_._1) // FIXME: add groupByKey
      .mapValues(_.map(_._2).toList)
      .map(identity)

  def dadOf: Map[Int, Int] = trios.flatMap{ t => t.dad.map(s => (t.kid, s)) }.toMap
  def momOf: Map[Int, Int] = trios.flatMap{ t => t.mom.map(s => (t.kid, s)) }.toMap
  def famOf: Map[Int, String] = trios.flatMap{ t => t.fam.map(s => (t.kid, s)) }.toMap

  def nSatisfying(filters: (Trio => Boolean)*): Int = trios.count(t => filters.forall(_(t)) )
  def nFam: Int = trios.flatMap(_.fam).toSet.size  // FIXME: add distinct
  def nIndiv: Int = trios.size
  def nCompleteTrio: Int = nSatisfying(_.isComplete)

  def writeSummary(filename: String) = {
    val columns = List(
      ("nFam", nFam), ("nIndiv", nIndiv), ("nCompleteTrios", nCompleteTrio),
      ("nMale", nSatisfying(_.isMale)), ("nFemale", nSatisfying(_.isFemale)),
      ("nCase", nSatisfying(_.isCase)), ("nControl", nSatisfying(_.isControl)),
      ("nMaleTrio", nSatisfying(_.isComplete, _.isMale)),
      ("nFemaleTrio", nSatisfying(_.isComplete, _.isFemale)),
      ("nCaseTrio", nSatisfying(_.isComplete, _.isCase)),
      ("nControlTrio", nSatisfying(_.isComplete, _.isControl)),
      ("nCaseMaleTrio", nSatisfying(_.isComplete, _.isCase, _.isMale)),
      ("nCaseFemaleTrio", nSatisfying(_.isComplete, _.isCase, _.isFemale)),
      ("nControlMaleTrio", nSatisfying(_.isComplete, _.isControl, _.isMale)),
      ("nControlFemaleTrio", nSatisfying(_.isComplete, _.isControl, _.isFemale)))

    withFileWriter(filename){ fw =>
      fw.write(columns.map(_._1).mkString("\t") + "\n")
      fw.write(columns.map(_._2).mkString("\t") + "\n")
    }
  }

  // FIXME: no header in plink fam file, but "FID\tKID\tPAT\tMAT\tSEX\tPHENO" sure seems appropriate
  def write(filename: String, sampleIds: Array[String]) {
    def sampleIdOrElse(s: Option[Int]) = if (s.isDefined) sampleIds(s.get) else "0"
    def toLine(t: Trio): String =
      t.fam.getOrElse("0") + "\t" + sampleIds(t.kid) + "\t" + sampleIdOrElse(t.dad) + "\t" +
        sampleIdOrElse(t.mom) + "\t" + t.sex.getOrElse("0") + "\t" + t.pheno.getOrElse("0") + "\n"
    val lines = trios.map(toLine)
    writeTable(filename, lines)
  }
}
