package org.broadinstitute.hail.driver

import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.expr.{EvalContext, Parser, TBoolean, TDouble, TGenotype, TSample, TString, Type}
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.io.plink.ExportBedBimFam
import org.kohsuke.args4j.{Option => Args4jOption}

object ExportPlink extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "Output file base (will generate .bed, .bim, .fam)")
    var output: String = _

    @Args4jOption(name = "-f", aliases = Array("--fam-expr"),
      usage = "Expression for .fam file values, in sample context only (global, s, sa in scope), assignable fields: famID, id, matID, patID (String), isFemale (Boolean), isCase (Boolean) or qPheno (Double)")
    var famExpr: String = "id = s.id"
  }

  def newOptions = new Options

  def name = "exportplink"

  def description = "Write current dataset as .bed/.bim/.fam"

  def supportsMultiallelic = false

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val ec = EvalContext(Map(
      "s" -> (0, TSample),
      "sa" -> (1, vds.saSignature),
      "global" -> (2, vds.globalSignature)))

    ec.set(2, vds.globalAnnotation)

    type Formatter = (() => Option[Any]) => () => String

    val formatID: Formatter = f => () => f().map(_.asInstanceOf[String]).getOrElse("0")
    val formatIsFemale: Formatter = f => () => f().map {
      _.asInstanceOf[Boolean] match {
        case true => "2"
        case false => "1"
      }
    }.getOrElse("0")
    val formatIsCase: Formatter = f => () => f().map {
      _.asInstanceOf[Boolean] match {
        case true => "2"
        case false => "1"
      }
    }.getOrElse("-9")
    val formatQPheno: Formatter = f => () => f().map(_.toString).getOrElse("-9")

    val famColumns: Map[String, (Type, Int, Formatter)] = Map(
      "famID" -> (TString, 0, formatID),
      "id" -> (TString, 1, formatID),
      "patID" -> (TString, 2, formatID),
      "matID" -> (TString, 3, formatID),
      "isFemale" -> (TBoolean, 4, formatIsFemale),
      "qPheno" -> (TDouble, 5, formatQPheno),
      "isCase" -> (TBoolean, 5, formatIsCase))

    val exprs = Parser.parseNamedExprs(options.famExpr, ec)

    val famFns: Array[() => String] = Array(
      () => "0", () => "0", () => "0", () => "0", () => "-9", () => "-9")

    exprs.foreach { case (name, t, f) =>
      famColumns.get(name) match {
        case Some((colt, i, formatter)) =>
          if (colt != t)
            fatal("invalid type for .fam file column $h: expected $colt, got $t")
          famFns(i) = formatter(f)

        case None =>
          fatal(s"no .fam file column $name")
      }
    }

    val spaceRegex = """\s+""".r
    val badSampleIds = vds.sampleIds.filter(id => spaceRegex.findFirstIn(id).isDefined)
    if (badSampleIds.nonEmpty) {
      fatal(
        s"""Found ${ badSampleIds.length } sample IDs with whitespace
            |  Please run `renamesamples' to fix this problem before exporting to plink format
            |  Bad sample IDs: @1 """.stripMargin, badSampleIds)
    }

    val bedHeader = Array[Byte](108, 27, 1)

    val plinkRDD = vds.rdd
      .mapValuesWithKey { case (v, (va, gs)) => ExportBedBimFam.makeBedRow(gs) }
      .persist(StorageLevel.MEMORY_AND_DISK)

    plinkRDD.map { case (v, bed) => bed }
      .saveFromByteArrays(options.output + ".bed", header = Some(bedHeader))

    plinkRDD.map { case (v, bed) => ExportBedBimFam.makeBimRow(v) }
      .writeTable(options.output + ".bim")

    plinkRDD.unpersist()

    val famRows = vds
      .sampleIdsAndAnnotations
      .map { case (s, sa) =>
        ec.setAll(s, sa)
        famFns.map(_()).mkString("\t")
      }

    state.hadoopConf.writeTextFile(options.output + ".fam")(out =>
      famRows.foreach(line => {
        out.write(line)
        out.write("\n")
      }))

    state
  }
}