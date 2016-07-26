package org.broadinstitute.hail.driver

import org.apache.spark.RangePartitioner
import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Querier
import org.broadinstitute.hail.io.annotators.SampleFamAnnotator
import org.broadinstitute.hail.methods.ExportBedBimFam
import org.broadinstitute.hail.variant.{Variant, VariantDataset}
import org.kohsuke.args4j.{Option => Args4jOption}
import org.broadinstitute.hail.expr.{Parser, TBoolean, TDouble, TNumeric, TString, TInt}

object ExportPlink extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "Output file base (will generate .bed, .bim, .fam)")
    var output: String = _

    @Args4jOption(required = false, name = "--rFam",
      usage = "If set, exports all (famID, patID, matID, sex, phenotype) fam information from the samples annotation." +
        "It expects the same schema as produced by annotatesamples fam. " +
        "The location of the annotations can be otherwise specified using " +
        "--rFamID, --rPID, --rMID, --rSex and --rPheno.")
    var rFam: String = _

    @Args4jOption(required = false, name = "--rFamID",
      usage = "If set, exports the family ID annotation in the fam file." +
        "Should contain the path to the Family ID annotation in the sample annotation. Needs to be a String.")
    var rFamID: String = _

    @Args4jOption(required = false, name = "--rPatID",
      usage = "If set, exports the paternal ID annotation in the fam file." +
        "Should contain the path to the Paternal ID annotation in the sample annotation. Needs to be a String.")
    var rPatID: String = _

    @Args4jOption(required = false, name = "--rMatID", aliases = Array("--maternalID-root"),
      usage = "If set, exports the maternal ID annotation in the fam file." +
        "Should contain the path to the Maternal ID annotation in the sample annotation.  Needs to be a String.")
    var rMatID: String = _

    @Args4jOption(required = false, name = "--rSex", aliases = Array("--sex-root"),
      usage = "If set, exports the sex annotation in the fam file." +
        "Should contain the path of sex annotation in the sample annotation.  " +
        "Needs to be an Int with the following values: 0 = unknown, 1 = male, 2 = female")
    var rSex: String = _

    @Args4jOption(required = false, name = "--rPheno", aliases = Array("--phenotype-root"),
      usage = "If set, exports the family ID annotation in the fam file." +
        "Should contain the path to the phenotype annotation in the sample annotation. Needs to be a numeric " +
        "(For case/control phenotypes, plink expects 0 = unknown, 1 = unaffected, 2 = affected).")
    var rPheno: String = _

  }
  val defaultFam = "0"
  val defaultPatID = "0"
  val defaultMatID = "0"
  val defaultSex = "0"
  val defaultPheno = "-9"

  def newOptions = new Options

  def name = "exportplink"

  def description = "Write current dataset as .bed/.bim/.fam"

  def supportsMultiallelic = false

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val spaceRegex = """\s+""".r
    val badSampleIds = vds.sampleIds.filter(id => spaceRegex.findFirstIn(id).isDefined)
    if (badSampleIds.nonEmpty) {
      val msg =
        s"""Found ${badSampleIds.length} sample IDs with whitespace.  Please run `renamesamples' to fix this problem before exporting to plink format.""".stripMargin
      log.error(msg + s"\n  Bad sample IDs: \n  ${badSampleIds.mkString("  \n")}")
      fatal(msg + s"\n  Bad sample IDs: \n  ${badSampleIds.take(10).mkString("  \n")}${
        if (badSampleIds.length > 10) "\n  ...\n  See hail.log for full list of IDs" else ""}")
    }

    val bedHeader = Array[Byte](108, 27, 1)

    val plinkVariantRDD = vds
      .rdd
      .map {
        case (v, va, gs) =>
          (v, ExportBedBimFam.makeBedRow(gs))
      }

    plinkVariantRDD.persist(StorageLevel.MEMORY_AND_DISK)

    val sortedPlinkRDD = plinkVariantRDD
      .repartitionAndSortWithinPartitions(new RangePartitioner[Variant, Array[Byte]]
      (vds.rdd.partitions.length, plinkVariantRDD))

    sortedPlinkRDD
      .persist(StorageLevel.MEMORY_AND_DISK)

    sortedPlinkRDD.map { case (v, bed) => bed }
      .saveFromByteArrays(options.output + ".bed", header = Some(bedHeader))

    sortedPlinkRDD.map { case (v, bed) => ExportBedBimFam.makeBimRow(v) }
      .writeTable(options.output + ".bim")

    plinkVariantRDD.unpersist()
    sortedPlinkRDD.unpersist()

    val famQuerier = Option(options.rFamID)
      .orElse(Option(options.rFam).map(code => options.rFam + "." + SampleFamAnnotator.famID ))
      .map(code => {
            val q = vds.querySA(code)
            q._1 match{
              case TString =>
              case _ => fatal(s"Family ID annotation $code was not of type String")
            }
        (si: Int) => q._2(vds.sampleAnnotations(si)).getOrElse(defaultFam).toString
      }).getOrElse((si: Int) => defaultFam)

    val patIDQuerier = Option(options.rPatID)
      .orElse(Option(options.rFam).map(code => options.rFam + "." + SampleFamAnnotator.patID ))
      .map(code => {
        val q = vds.querySA(code)
        q._1 match{
          case TString =>
          case _ => fatal(s"Paternal ID annotation $code was not of type String")
        }
        (si: Int) => q._2(vds.sampleAnnotations(si)).getOrElse(defaultPatID).toString
      }).getOrElse((si: Int) => defaultPatID)

    val matIDQuerier = Option(options.rMatID)
      .orElse(Option(options.rFam).map(code => options.rFam + "." + SampleFamAnnotator.matID ))
      .map(code => {
        val q = vds.querySA(code)
        q._1 match{
          case TString =>
          case _ => fatal(s"Maternal ID annotation $code was not of type String")
        }
        (si: Int) => q._2(vds.sampleAnnotations(si)).getOrElse(defaultMatID).toString
      }).getOrElse((si: Int) => defaultMatID)

    val sexQuerier = if(options.rSex != null){
      val q = vds.querySA(options.rSex)
      q._1 match{
        case TInt =>
        case _ => fatal("Sample sex annotation %s was not of type Int".format(options.rSex))
      }
      (si: Int) => {
        val sex = q._2(vds.sampleAnnotations(si)).getOrElse(defaultSex).asInstanceOf[Int]
        if (sex < 0 || sex > 2) {
          fatal(s"Value $sex is not an acceptable value for sample sex annotation.")
        }
        sex.toString
      }
    } else{
      Option(options.rFam)
      .map(c => {
        val q = vds.querySA(c + "." + SampleFamAnnotator.isFemale)
        q._1 match{
          case TBoolean =>
          case _ => fatal("Sample sex annotation (from default fam path) %s was not of type Boolean"
            .format(c + "." + SampleFamAnnotator.isFemale))
        }
        (si: Int) => {
          q._2(vds.sampleAnnotations(si)) match{
            case Some(isFemale) => if(isFemale.asInstanceOf[Boolean]) "2" else "1"
            case None => defaultSex
          }
        }
      }).getOrElse((si: Int) => defaultSex)
    }

    val phenoQuerier = if(options.rPheno != null){
      val q = vds.querySA(options.rPheno)
      if(!q._1.isInstanceOf[TNumeric]){
        fatal("Sample phenotype annotation %s was not found or is not of type Numeric".format(options.rPheno))
      }
      (si: Int) =>  q._2(vds.sampleAnnotations(si)).getOrElse(defaultPheno).toString
    } else{
      Option(options.rFam)
        .map(code => {
          if (vds.saSignature.getOption(Parser.parseAnnotationRoot(code + "." + SampleFamAnnotator.qPheno,"sa"):_*).isDefined) {
            val q = vds.querySA(code + "." + SampleFamAnnotator.qPheno)
            q._1 match{
              case TDouble =>
              case _ => fatal("Quantitative phenotype annotation (from default fam path) %s was not of type Double"
                .format(code + "." + SampleFamAnnotator.qPheno))
            }
            (si: Int) =>  vds.querySA(code + "." + SampleFamAnnotator.qPheno)
              ._2(vds.sampleAnnotations(si)).getOrElse(defaultPheno).toString
          }
          else if(vds.saSignature.getOption(Parser.parseAnnotationRoot(code + "." + SampleFamAnnotator.isCase,"sa"):_*).isDefined){
            val q = vds.querySA(code + "." + SampleFamAnnotator.isCase)
            q._1 match{
              case TBoolean =>
              case _ => fatal("Case/control phenotype annotation (from default fam path) %s was not of type Boolean"
                .format(code + "." + SampleFamAnnotator.isCase))
            }
            (si: Int) => {
              q._2(vds.sampleAnnotations(si)) match{
                case Some(isCase) => if(isCase.asInstanceOf[Boolean]) "2" else "1"
                case None => defaultPheno
              }
            }
          }
          else{
            fatal("Sample phenotype annotation could not be found. " +
              "Paths evaluated (based on default fam): %s [Double], %s [Boolean]"
                .format(code + "." + SampleFamAnnotator.qPheno,
                  code + "." + SampleFamAnnotator.isCase))
            (si: Int) => defaultPheno
          }
        }).getOrElse((si: Int) => defaultPheno)
    }


    val famRows = vds
      .sampleIds
      .zipWithIndex
      .map({
        case(s,si) => ExportBedBimFam.makeFamRow(
          famQuerier(si),
          s,
          patIDQuerier(si),
          matIDQuerier(si),
          sexQuerier(si),
          phenoQuerier(si)
        )
      })

    writeTextFile(options.output + ".fam", state.hadoopConf)(oos =>
      famRows.foreach(line => {
        oos.write(line)
        oos.write("\n")
      }))

    state
  }
}