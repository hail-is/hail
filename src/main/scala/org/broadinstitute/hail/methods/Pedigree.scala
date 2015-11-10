package org.broadinstitute.hail.methods

import java.io.File
import org.apache.hadoop
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.variant.{Sex, Phenotype}
import org.broadinstitute.hail.variant.Phenotype.{Phenotype, Case, Control}
import org.broadinstitute.hail.variant.Sex.{Sex, Male, Female}
import scala.io.Source

object Role extends Enumeration {
  type Role = Value
  val Kid = Value("0")
  val Dad = Value("1")
  val Mom = Value("2")
}

import org.broadinstitute.hail.methods.Role.{Role, Kid, Dad, Mom}

case class Trio(kid: Int, fam: Option[String], dad: Option[Int], mom: Option[Int],
                sex: Option[Sex], pheno: Option[Phenotype]) {

  def toCompleteTrio: Option[CompleteTrio] =
    dad.flatMap(d =>
      mom.map(m => CompleteTrio(kid, fam, d, m, sex, pheno)))

  def isMale: Boolean = sex.contains(Male)
  def isFemale: Boolean = sex.contains(Female)
  def isCase: Boolean = pheno.contains(Case)
  def isControl: Boolean = pheno.contains(Control)
  def isComplete: Boolean = dad.isDefined && mom.isDefined
}

case class CompleteTrio(kid: Int, fam: Option[String], dad: Int, mom: Int, sex: Option[Sex], pheno: Option[Phenotype])

object Pedigree {

  def read(filename: String, sampleIds: Array[String]): Pedigree = {
    require(filename.endsWith(".fam"))

    val sampleIndex: Map[String, Int] = sampleIds.zipWithIndex.toMap
    def maybeId(id: String): Option[Int] = if (id != "0") sampleIndex.get(id) else None
    def maybeFam(fam: String): Option[String] = if (fam != "0") Some(fam) else None

    Pedigree(Source.fromFile(new File(filename))
      .getLines()
      .filter(line => !line.isEmpty)
      .map { line => // FIXME: proper input error handling (and possible conflicting trio handing)
      val Array(fam, kid, dad, mom, sex, pheno) = line.split("\\s+")

      Trio(sampleIndex(kid), maybeFam(fam), maybeId(dad), maybeId(mom),
        Sex.withNameOption(sex), Phenotype.withNameOption(pheno))
      }
      .toArray
    )
  }
}

case class Pedigree(trios: Array[Trio]) {

  def completeTrios: Array[CompleteTrio] = trios.flatMap(_.toCompleteTrio)

  // plink only prints # of kids under CHLD, but the list of kids may be useful, currently not used anywhere else
  def nuclearFams: Map[(Int, Int), Iterable[Int]] =
    completeTrios.map(t => ((t.dad, t.mom), t.kid)).toMap.groupByKey

  def nSatisfying(filters: (Trio => Boolean)*): Int = trios.count(t => filters.forall(_(t)) )

  def writeSummary(filename: String, hConf: hadoop.conf.Configuration) = {
    val columns = List(
      ("nIndiv", trios.length), ("nCompleteTrios", completeTrios.length), ("nNuclearFams", nuclearFams.size),
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

    writeTextFile(filename, hConf){ fw =>
      fw.write(columns.map(_._1).mkString("\t") + "\n")
      fw.write(columns.map(_._2).mkString("\t") + "\n")
    }
  }

  // plink does not print a header in .mendelf, but "FID\tKID\tPAT\tMAT\tSEX\tPHENO" seems appropriate
  def write(filename: String, hConf: hadoop.conf.Configuration, sampleIds: Array[String]) {
    def sampleIdOrElse(s: Option[Int]) = if (s.isDefined) sampleIds(s.get) else "0"
    def toLine(t: Trio): String =
      t.fam.getOrElse("0") + "\t" + sampleIds(t.kid) + "\t" + sampleIdOrElse(t.dad) + "\t" +
        sampleIdOrElse(t.mom) + "\t" + t.sex.getOrElse("0") + "\t" + t.pheno.getOrElse("0") + "\n"
    val lines = trios.map(toLine)
    writeTable(filename, hConf, lines)
  }
}
