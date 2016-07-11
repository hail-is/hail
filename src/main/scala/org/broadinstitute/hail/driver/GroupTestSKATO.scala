package org.broadinstitute.hail.driver

import org.apache.spark.sql.Row
import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.annotations.Annotation
import org.json4s._
import org.kohsuke.args4j.{Option => Args4jOption}
import org.broadinstitute.hail.Utils._
import org.json4s.jackson.JsonMethods._
import java.io.{FileInputStream, IOException}
import java.util.Properties
import scala.collection.JavaConverters._
import scala.util.Random

object GroupTestSKATO extends Command {

  class Options extends BaseOptions {

    @Args4jOption(required = true, name = "-k", aliases = Array("--group-keys"),
      usage = "comma-separated list of annotations to be used as grouping variable(s) (must be attribute of va)")
    var groupKeys: String = _

    @Args4jOption(required = true, name = "--config", usage = "SKAT-O configuration file")
    var config: String = _

    @Args4jOption(name = "--block-size", usage = "# of Groups per SKAT-O invocation")
    var blockSize = 1000

    @Args4jOption(required = false, name = "-q", aliases = Array("--quantitative"), usage = "y is a quantitative phenotype")
    var quantitative: Boolean = false

    @Args4jOption(required = true, name = "-y", aliases = Array("--y"), usage = "Response sample annotation")
    var ySA: String = _

    @Args4jOption(required = false, name = "-c", aliases = Array("--covariates"), usage = "Covariate sample annotations, comma-separated")
    var covSA: String = ""

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "path of output tsv")
    var output: String = _

    @Args4jOption(required = false, name = "--seed", usage = "number to set random seed to [default=1]")
    var seed: Int = 1

    @Args4jOption(required = false, name = "--random-seed", usage = "use a random seed")
    var randomize: Boolean = false


    //SKAT_Null_Model options
    @Args4jOption(required = false, name = "--n-resampling",
      usage = "Number of times to resample residuals")
    var nResampling: Int = 0

    @Args4jOption(required = false, name = "--type-resampling",
      usage = "Resampling method. One of [bootstrap, bootstrap.fast]")
    var typeResampling: String = "bootstrap"

    @Args4jOption(required = false, name = "--no-adjustment",
      usage = "No adjustment for small sample sizes")
    var noAdjustment: Boolean = false


    //SKAT options
    @Args4jOption(required = false, name = "--kernel",
      usage = "SKAT-O kernel type. One of [linear, linear.weighted, IBS, IBS.weighted, quadratic, 2wayIX]")
    var kernel: String = "linear.weighted"

    @Args4jOption(required = false, name = "--method",
      usage = "Method for calculating p-values. One of [davies, liu, liu.mod, optimal.adj, optimal]")
    var method: String = "davies"

    @Args4jOption(required = false, name = "--weights-beta",
      usage = "Comma-separated parameters for beta function for calculating weights. Default is 1,25")
    var weightsBeta: String = "1,25"

    @Args4jOption(required = false, name = "--impute-method",
      usage = "Method for imputing missing genotypes. One of [fixed, random, bestguess]")
    var imputeMethod: String = "fixed"

    @Args4jOption(required = false, name = "--r-corr",
      usage = "The rho parameter for the unified test. rho=0 is SKAT only. rho=1 is Burden only. Can also be multiple comma-separated values")
    var rCorr: String = "0.0"

    @Args4jOption(required = false, name = "--missing-cutoff",
      usage = "Missing rate cutoff for variant inclusion")
    var missingCutoff: Double = 0.15

    @Args4jOption(required = false, name = "--estimate-maf",
      usage = "Method for estimating MAF. 1 = use all samples to estimate MAF. 2 = use samples with non-missing phenotypes and covariates")
    var estimateMAF: Int = 1

  }

  def newOptions = new Options

  def name = "grouptest skato"

  def description = "Run SKAT-O on groups"

  def supportsMultiallelic = false

  def requiresVDS = true

  val skatoSignature = TArray(TStruct(
    "groupName" -> TString,
    "pValue" -> TDouble,
    "pValueNoAdj" -> TDouble,
    "nMarker" -> TInt,
    "nMarkerTest" -> TInt
  ))

  val header = "groupName\tpValue\tpValueNoAdj\tnMarker\tnMarkerTest"

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val sampleIds = vds.sampleIds

    val properties = try {
      val p = new Properties()
      val is = new FileInputStream(options.config)
      p.load(is)
      is.close()
      p
    } catch {
      case e: IOException =>
        fatal(s"could not open file: ${e.getMessage}")
    }

    val yType = if (options.quantitative) "C" else "D"
    val nResampling = options.nResampling
    val typeResampling = options.typeResampling
    val adjustment = !options.noAdjustment
    val kernel = options.kernel
    val method = options.method
    val weightsBeta = options.weightsBeta
    val imputeMethod = options.imputeMethod
    val rCorr = options.rCorr
    val missingCutoff = options.missingCutoff
    val estimateMAF = options.estimateMAF
    val quantitative = options.quantitative

    val R = properties.getProperty("hail.skato.Rscript", "/usr/bin/Rscript")
    val skatoScript = properties.getProperty("hail.skato.script", "src/dist/scripts/skato.r")

    val rCorrArray = options.rCorr.split(",").map(_.toDouble)
    if (!rCorrArray.forall(d => d >= 0.0 && d <= 1.0))
      fatal(s"Arguments to --r-corr must be numeric, between 0.0 and 1.0, and comma-separated. Found ${options.rCorr}")

    val weightBetaArray = options.weightsBeta.split(",").map(_.toDouble)
    if (weightBetaArray.length != 2)
      fatal(s"Must give two comma-separated arguments to --weights-beta. Found ${options.weightsBeta}")

    val availMethods = Set("davies", "liu", "liu.mod", "optimal.adj", "optimal")
    if (!availMethods.contains(method))
      fatal("Did not recognize option specified for --method. Use one of [davies, liu, liu.mod, optimal.adj, optimal]")

    val availKernels = Set("linear", "linear.weighted", "IBS", "IBS.weighted", "quadratic", "2wayIX")
    if (!availKernels.contains(kernel))
      fatal("Did not recognize option specified for --kernel. Use one of [linear, linear.weighted, IBS, IBS.weighted, quadratic, 2wayIX]")

    val availMafEstimate = Set(1, 2)
    if (!availMafEstimate.contains(estimateMAF))
      fatal("Did not recognize option specified for --estimate-maf. Use one of [1, 2].")

    val availResamplingTypes = Set("bootstrap", "bootstrap.fast", "permutation")
    if (!availResamplingTypes.contains(typeResampling))
      fatal("Did not recognize option specified for --type-resampling. Use one of [bootstrap, permutation, bootstrap.fast]. bootstrap.fast only available for non-quantitative variables")
    if (quantitative && typeResampling == "bootstrap.fast")
      fatal("Can only use bootstrap.fast for --type-resampling if -q is not set (dichotomous variables only).")

    if (!(rCorrArray.length == 1 && rCorrArray(0) == 0.0) && kernel != "linear" && kernel != "linear.weighted")
      fatal("Non-zero r.corr only can be used with linear or linear.weighted kernels")

    def toDouble(t: BaseType, code: String): Any => Double = t match {
      case TInt => _.asInstanceOf[Int].toDouble
      case TLong => _.asInstanceOf[Long].toDouble
      case TFloat => _.asInstanceOf[Float].toDouble
      case TDouble => _.asInstanceOf[Double]
      case TBoolean => _.asInstanceOf[Boolean].toDouble
      case _ => fatal(s"Sample annotation `$code' must be numeric or Boolean, got $t")
    }

    val queriers = options.groupKeys.split(",").map{vds.queryVA(_)._2}

    val symTab = Map(
      "s" -> (0, TSample),
      "sa" -> (1, vds.saSignature))

    val ec = EvalContext(symTab)
    val a = ec.a

    val (yT, yQ) = Parser.parse(options.ySA, ec)

    val yToDouble = toDouble(yT, options.ySA)
    val ySA = vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      a(0) = s
      a(1) = sa
      yQ().map(yToDouble)
    }

    val (covT, covQ) = Parser.parseExprs(options.covSA, ec).unzip
    val covToDouble = (covT, options.covSA.split(",").map(_.trim)).zipped.map(toDouble)
    val covSA = vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      a(0) = s
      a(1) = sa
      (covQ.map(_()), covToDouble).zipped.map(_.map(_))
    }
    val nCovar = covT.size

    val (yDefined, covDefined, samplesDefined) =
      (ySA, covSA, sampleIds)
        .zipped
        .filter( (y, c, s) => y.isDefined && c.forall(_.isDefined))

    if (samplesDefined.length < 1) {
      val sb = new StringBuilder()
      sb.append(s"No samples with non-missing phenotype [${options.ySA}]")
      if (nCovar > 0)
        sb.append(s" and covariates [${options.covSA}]")
      fatal(sb.result())
    }
    info(s"${samplesDefined.length} samples with non-missing phenotypes")

    val seed = if (options.randomize) Random.nextInt() else options.seed
    info(s"Using a random seed of [$seed]")

    val cmd = Array(R, skatoScript)

    val localBlockSize = options.blockSize

    val ySABC = state.sc.broadcast(ySA)
    val covSABC = state.sc.broadcast(covSA)

    def printContext(w: (String) => Unit) = {
      val y = pretty(JArray(ySABC.value.map{y => if (y.isDefined) JDouble(y.get) else JNull}.toList))
      val cov = pretty(JArray(covSABC.value.map{cov => JArray(cov.map{c => if (c.isDefined) JDouble(c.get) else JNull}.toList)}.toList))
      val covString = if (nCovar > 0) s""""COV":$cov,""" else ""
      w( s"""{"yType":"$yType",
             |"nResampling":$nResampling,
             |"typeResampling":"$typeResampling",
             |"adjustment":$adjustment,
             |"kernel":"$kernel",
             |"method":"$method",
             |"weightsBeta":[$weightsBeta],
             |"imputeMethod":"$imputeMethod",
             |"rCorr":[$rCorr],
             |"missingCutoff":$missingCutoff,
             |"estimateMAF":$estimateMAF,
             |"seed":$seed,
             |"Y":$y,
             |$covString
             |"groups":{""".stripMargin)
    }

    def printSep(w: (String) => Unit) = {
      w(",")
    }

    def printFooter(w: (String) => Unit) = {
      w("}}")
    }

    def printElement(w: (String) => Unit, g: (IndexedSeq[Any], Iterable[Iterable[Int]])) = {
      val a = JArray(g._2.map(a => JArray(a.map(JInt(_)).toList)).toList)
      w(s""""${g._1.map(_.toString).mkString(",")}":${pretty(a)}""")
    }

    val groups = vds.rdd.map{case (v, va, gs) =>
      (queriers.map{_(va).get}.toIndexedSeq, gs.map{g => g.nNonRefAlleles.getOrElse(9)}) //FIXME: Expand out arrays
    }.groupByKey().persist(StorageLevel.MEMORY_AND_DISK) // FIXME: Remove None/null groups and duplicate variants

    groups.mapPartitions{ it =>
      val pb = new java.lang.ProcessBuilder(cmd.toList.asJava)

      it.grouped(localBlockSize)
        .flatMap{_.iterator
          .pipe(pb, printContext, printElement, printFooter, printSep)
          .map{result =>
            val a = Annotation.fromJson(parse(result), skatoSignature, "<root>")
            a.asInstanceOf[IndexedSeq[Any]].map(_.asInstanceOf[Row].mkString("\t")).mkString("\n")
          }
        }
    }.writeTable(options.output, Some(header), deleteTmpFiles = true)

    groups.unpersist()

    state
  }
}
