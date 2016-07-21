package org.broadinstitute.hail.driver

import org.apache.spark.RangePartitioner
import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Querier
import org.broadinstitute.hail.io.annotators.SampleFamAnnotator
import org.broadinstitute.hail.methods.ExportBedBimFam
import org.broadinstitute.hail.variant.{Variant, VariantDataset}
import org.kohsuke.args4j.{Option => Args4jOption}
import org.broadinstitute.hail.expr.{Parser,TBoolean}

object ExportPlink extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "Output file base (will generate .bed, .bim, .fam)")
    var output: String = _

    @Args4jOption(required = false, name = "-r", aliases = Array("--fam-root"),
      usage = "Root of fam annotation in the samples. Annotations are expected to be in the same structure as created by annotatesamples fam. If not, field can be overriden with -r* options.")
    var fr: String = ""

    @Args4jOption(required = false, name = "-rFamID", aliases = Array("--famID-root"),
      usage = "Root of Family ID annotation in the sample annotation. No need to provide if -r is used.")
    var rFamID: String = ""

    @Args4jOption(required = false, name = "-rPID", aliases = Array("--paternalID-root"),
      usage = "Root of Paternal ID annotation in the sample annotation. No need to provide if -r is used.")
    var rPatID: String = ""

    @Args4jOption(required = false, name = "-rMID", aliases = Array("--maternalID-root"),
      usage = "Root of Maternal ID annotation in the sample annotation. No need to provide if -r is used.")
    var rMatID: String = ""

    @Args4jOption(required = false, name = "-rSex", aliases = Array("--sex-root"),
      usage = "Root of sex annotation in the sample annotation. No need to provide if -r is used.")
    var rSex: String = ""

    @Args4jOption(required = false, name = "-rPheno", aliases = Array("--phenotype-root"),
      usage = "Root of sex annotation in the sample annotation. No need to provide if -r is used.")
    var rPheno: String = ""

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

  def getFamRows(vds: VariantDataset, options: Options) : IndexedSeq[String] = {

    class famValue(val path: String, val defaultValue : String) {
      //get Querier
      val querier = if(path.isEmpty) None else Some(vds.querySA(path))
      def getValue(si: Int) : String = {
        querier match{
          case Some(q) => q._2(vds.sampleAnnotations(si)).getOrElse(defaultValue).toString
          case None => defaultValue
        }
      }
    }

    class sexValue(path: String) extends famValue(path, defaultSex){
      override def getValue(si: Int) = {
        querier match {
          case Some(q) =>
            q._2(vds.sampleAnnotations(si)) match {
              case Some(isF) => if(isF.asInstanceOf[Boolean]) "2" else "1"
              case None => defaultValue
            }
          case None => defaultValue
        }
      }
    }

    class phenoValue(path: String) extends famValue(path, defaultPheno){
      override def getValue(si: Int) = {
        querier match {
          case Some(q) =>
            q._2(vds.sampleAnnotations(si)) match {
              case Some(pheno) =>
                q._1 match {
                  case TBoolean => if (pheno.asInstanceOf[Boolean]) "2" else "1"
                  case _ => pheno.toString
                }
              case None => defaultValue
            }
          case None => defaultValue
        }
      }
    }

    def getPath(famRoot: String, fieldPath: String, defaultLeaf: String*) : String = {
      if(fieldPath.isEmpty){
        if(famRoot.isEmpty){
          ""
        }else{
          defaultLeaf.foldLeft("")(
            (path,str) => if(vds.saSignature.getOption(Parser.parseAnnotationRoot(famRoot + "." + str,"sa")).isDefined) famRoot + "." + str else path
          )
        }
      }else{
        fieldPath
      }
    }

    val famID = new famValue(getPath(options.fr,options.rFamID,SampleFamAnnotator.famID),defaultFam)
    val patID = new famValue(getPath(options.fr,options.rPatID,SampleFamAnnotator.patID),defaultPatID)
    val matID= new famValue(getPath(options.fr,options.rMatID,SampleFamAnnotator.matID),defaultMatID)
    val sex = new sexValue(getPath(options.fr,options.rSex,SampleFamAnnotator.isFemale))
    val pheno = new phenoValue(getPath(options.fr,options.rPheno,SampleFamAnnotator.isCase,SampleFamAnnotator.qPheno))

      vds
        .sampleIds
        .zipWithIndex
        .map({
          case(s,si) => ExportBedBimFam.makeFamRow(
            famID.getValue(si),
            s,
            patID.getValue(si),
            matID.getValue(si),
            sex.getValue(si),
            pheno.getValue(si)
          )
        })

  }

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

    if(!options.fr.isEmpty){

    }
    
    val famRows = getFamRows(vds,options)

    writeTextFile(options.output + ".fam", state.hadoopConf)(oos =>
      famRows.foreach(line => {
        oos.write(line)
        oos.write("\n")
      }))

    state
  }
}