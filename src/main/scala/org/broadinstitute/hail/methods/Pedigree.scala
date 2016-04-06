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
    if (!filename.endsWith(".fam"))
      fatal("-f | --fam filename must end in .fam")

    val sampleIndex: Map[String, Int] = sampleIds.zipWithIndex.toMap

    var nSamplesDiscarded = 0

    // .fam samples not in sampleIds are discarded
    readFile(filename, hConf) { s =>
      val sampleSet = collection.mutable.Set[Int]()

      val trios = Source.fromInputStream(s)
        .getLines()
        .filter(line => !line.isEmpty)
        .flatMap{ line => // FIXME: check that pedigree makes sense (e.g., cannot be own parent)
          val splitLine = line.split("\\s+")  // FIXME: fails on names with spaces, will fix in PR for adding .fam to annotations by giving delimiter option
          fatalIf(splitLine.size != 6, s"Require 6 fields per line in .fam, but this line has ${splitLine.size}: $line")
          val Array(fam, kid, dad, mom, sex, pheno) = splitLine
          sampleIndex.get(kid) match {
            case Some(s) =>
              if (sampleSet(s))
                fatal(s".fam sample name is not unique: $kid")
              else
                sampleSet += s
              Some(Trio(
                s,
                if (fam != "0") Some(fam) else None,
                if (dad != "0") sampleIndex.get(dad) else None,
                if (mom != "0") sampleIndex.get(mom) else None,
                Sex.withNameOption(sex),
                Phenotype.withNameOption(pheno)))
            case None =>
              nSamplesDiscarded += 1
              None
          }
        }.toArray

      if (nSamplesDiscarded > 0)
        warn(s"$nSamplesDiscarded ${plural(nSamplesDiscarded, "sample")} discarded from .fam: missing from variant data set.")

      Pedigree(trios)
    }
  }

  // plink only prints # of kids under CHLD, but the list of kids may be useful, currently not used anywhere else
  def nuclearFams(completeTrios: Array[CompleteTrio]): Map[(Int, Int), Array[Int]] =
    completeTrios.groupBy(t => (t.dad, t.mom)).mapValues(_.map(_.kid)).force
}

case class Pedigree(trios: Array[Trio]) {
  
  def completeTrios: Array[CompleteTrio] = trios.flatMap(_.toCompleteTrio)

  def samplePheno: Map[Int, Option[Phenotype]] = trios.iterator.map(t => (t.kid, t.pheno)).toMap

  def phenotypedSamples: Set[Int] = trios.filter(_.pheno.isDefined).map(_.kid).toSet

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
        sampleIdOrElse(t.mom) + "\t" + t.sex.getOrElse("0") + "\t" + t.pheno.getOrElse("0")
    val lines = trios.map(toLine)
    writeTable(filename, hConf, lines)
  }
}
