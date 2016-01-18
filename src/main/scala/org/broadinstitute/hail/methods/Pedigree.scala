package org.broadinstitute.hail.methods

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

  def read(filename: String, hConf: hadoop.conf.Configuration, sampleIds: IndexedSeq[String]): Pedigree = {
    require(filename.endsWith(".fam"))

    val sampleIndex: Map[String, Int] = sampleIds.zipWithIndex.toMap

    // .fam samples not in sampleIds are discarded
    readFile(filename, hConf) { s =>
      Pedigree(Source.fromInputStream(s)
        .getLines()
        .filter(line => !line.isEmpty)
        .flatMap{ line => // FIXME: check that pedigree makes sense (e.g., cannot be own parent)
          val Array(fam, kid, dad, mom, sex, pheno) = line.split("\\s+")
          sampleIndex.get(kid).map( kidId =>
            Trio(
              kidId,
              if (fam != "0") Some(fam) else None,
              sampleIndex.get(dad), // FIXME: code assumes "0" cannot be a (string) sample ID in a vds, do you agree we should enforce that elsewhere?
              sampleIndex.get(mom),
              Sex.withNameOption(sex),
              Phenotype.withNameOption(pheno))
          )
        }.toArray
      )
    }
  }

  // plink only prints # of kids under CHLD, but the list of kids may be useful, currently not used anywhere else
  def nuclearFams(completeTrios: Array[CompleteTrio]): Map[(Int, Int), Array[Int]] =
    completeTrios.groupBy(t => (t.dad, t.mom)).mapValues(_.map(_.kid)).force
}

case class Pedigree(trios: Array[Trio]) {
  if (trios.map(_.kid).toSet.size != trios.size)
    fatal(".fam sample names are not unique.")

  def completeTrios: Array[CompleteTrio] = trios.flatMap(_.toCompleteTrio)

  def samplePheno: Map[Int, Option[Phenotype]] = trios.iterator.map(t => (t.kid, t.pheno)).toMap

  def nSatisfying(filters: (Trio => Boolean)*): Int = trios.count(t => filters.forall(_ (t)))

  def writeSummary(filename: String, hConf: hadoop.conf.Configuration) = {
    val columns = Array(
      ("nIndiv", trios.length), ("nTrios", completeTrios.length),
      ("nNuclearFams", Pedigree.nuclearFams(completeTrios).size),
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

    writeTextFile(filename, hConf) { fw =>
      fw.write(columns.iterator.map(_._1).mkString("\t") + "\n")
      fw.write(columns.iterator.map(_._2).mkString("\t") + "\n")
    }
  }

  // plink does not print a header in .mendelf, but "FID\tKID\tPAT\tMAT\tSEX\tPHENO" seems appropriate
  def write(filename: String, hConf: hadoop.conf.Configuration, sampleIds: IndexedSeq[String]) {
    def sampleIdOrElse(s: Option[Int]) = if (s.isDefined) sampleIds(s.get) else "0"
    def toLine(t: Trio): String =
      t.fam.getOrElse("0") + "\t" + sampleIds(t.kid) + "\t" + sampleIdOrElse(t.dad) + "\t" +
        sampleIdOrElse(t.mom) + "\t" + t.sex.getOrElse("0") + "\t" + t.pheno.getOrElse("0") + "\n"
    val lines = trios.map(toLine)
    writeTable(filename, hConf, lines)
  }
}
