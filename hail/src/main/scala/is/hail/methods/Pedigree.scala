package is.hail.methods

import is.hail.check.Gen
import is.hail.utils._
import is.hail.variant.Sex.{Female, Male, Sex}
import is.hail.variant.Sex
import org.apache.hadoop

import scala.collection.mutable
import scala.io.Source

object Role extends Enumeration {
  type Role = Value
  val Kid = Value("0")
  val Dad = Value("1")
  val Mom = Value("2")
}

trait Trio {
  def kid: String

  def fam: Option[String]

  def dad: Option[String]

  def mom: Option[String]

  def sex: Option[Sex]

  def isMale: Boolean = sex.contains(Male)

  def isFemale: Boolean = sex.contains(Female)

  def toCompleteTrio: Option[CompleteTrio]

  def isComplete: Boolean = toCompleteTrio.isDefined

  def restrictTo(ids: Set[String]): Option[Trio]
}

case class BaseTrio(kid: String, fam: Option[String], dad: Option[String], mom: Option[String],
  sex: Option[Sex]) extends Trio {

  def toCompleteTrio: Option[CompleteTrio] =
    dad.flatMap(d =>
      mom.map(m => CompleteTrio(kid, fam, d, m, sex)))

  def restrictTo(ids: Set[String]): Option[BaseTrio] = if (!ids(kid))
    None
  else
    Some(copy(
      dad = dad.filter(ids),
      mom = mom.filter(ids)
    ))
}

case class CompleteTrio(kid: String, fam: Option[String], knownDad: String, knownMom: String,
  sex: Option[Sex]) extends Trio {

  def dad: Option[String] = Some(knownDad)

  def mom: Option[String] = Some(knownMom)

  def toCompleteTrio = Some(this)

  def restrictTo(ids: Set[String]): Option[CompleteTrio] = if (!ids(kid) || !ids(knownDad) || !ids(knownMom))
    None
  else Some(this)
}

object Pedigree {

  def read(filename: String, hConf: hadoop.conf.Configuration, delimiter: String = "\\s+"): Pedigree = {
    hConf.readLines(filename) { lines =>

      val invalidSex = mutable.Set.empty[String]
      var filteredSex = 0
      val trios = lines.filter(line => !line.value.isEmpty)
        .map {
          _.map { line => // FIXME: check that pedigree makes sense (e.g., cannot be own parent)
            val splitLine = line.split(delimiter)
            if (splitLine.size != 6)
              fatal(s"Require 6 fields per line in .fam, but this line has ${ splitLine.size }: $line")
            val Array(fam, kid, dad, mom, sex, _) = splitLine
            val sexOption = Sex.withNameOption(sex)
            if (sexOption.isEmpty) {
              filteredSex += 1
              invalidSex += sex
            }
            BaseTrio(
              kid,
              if (fam != "0") Some(fam) else None,
              if (dad != "0") Some(dad) else None,
              if (mom != "0") Some(mom) else None,
              Sex.withNameOption(sex))
          }.value
        }.toArray
      val duplicates = trios.map(_.kid).duplicates()

      if (duplicates.nonEmpty)
        fatal(
          s"""Invalid pedigree: found duplicate proband IDs
             |  [ @1 ]""".stripMargin, duplicates)

      if (filteredSex > 0)
        warn(
          s"""Found $filteredSex samples with missing sex information (not 1 or 2).
             |  Missing sex identifiers: [ @1 ]""".stripMargin, invalidSex)

      Pedigree(trios)
    }
  }

  // plink only prints # of kids under CHLD, but the list of kids may be useful, currently not used anywhere else
  def nuclearFams(completeTrios: IndexedSeq[CompleteTrio]): Map[(String, String), IndexedSeq[String]] =
    completeTrios.groupBy(t => (t.knownDad, t.knownMom)).mapValues(_.map(_.kid)).force

  def gen(sampleIds: IndexedSeq[String]): Gen[Pedigree] = {
    Gen.parameterized { p =>
      val rng = p.rng
      Gen.shuffle(sampleIds)
        .map { is =>
          val groups = is.grouped(3)
            .filter(_ => rng.nextUniform(0, 1) > 0.25)
            .map { g =>
              val r = rng.nextUniform(0, 1)
              if (r < 0.10)
                g.take(1)
              if (r < 0.20)
                g.take(2)
              else
                g
            }
          val trios = groups.map { g =>
            val (kid, mom, dad) = (g(0),
              if (g.length >= 2) Some(g(1)) else None,
              if (g.length >= 3) Some(g(2)) else None)
            BaseTrio(kid, fam = None, mom = mom, dad = dad, sex = None)
          }
            .toArray
          new Pedigree(trios)
        }
    }
  }

  def genWithIds(): Gen[(IndexedSeq[String], Pedigree)] = {
    for (ids <- Gen.distinctBuildableOf(Gen.identifier);
      ped <- gen(ids))
      yield (ids, ped)
  }
}

case class Pedigree(trios: IndexedSeq[Trio]) {

  def filterTo(ids: Set[String]): Pedigree = copy(trios = trios.flatMap(_.restrictTo(ids)))

  def completeTrios: IndexedSeq[CompleteTrio] = trios.flatMap(_.toCompleteTrio)

  def nSatisfying(filters: (Trio => Boolean)*): Int = trios.count(t => filters.forall(_ (t)))

  // plink does not print a header in .mendelf, but "FID\tKID\tPAT\tMAT\tSEX\tPHENO" seems appropriate
  def write(filename: String, hConf: hadoop.conf.Configuration) {
    def sampleIdOrElse(s: Option[String]) = s.getOrElse("0")

    def toLine(t: Trio): String =
      t.fam.getOrElse("0") + "\t" + t.kid + "\t" + sampleIdOrElse(t.dad) + "\t" +
        sampleIdOrElse(t.mom) + "\t" + t.sex.getOrElse("0") + "\t0"

    val lines = trios.map(toLine)
    hConf.writeTable(filename, lines)
  }
}
