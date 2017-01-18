package is.hail.driver

import is.hail.io._
import is.hail.utils._
import is.hail.io.vcf.LoadVCF
import org.kohsuke.args4j.{Argument, Option => Args4jOption}

import scala.collection.JavaConverters._
import org.apache.hadoop
import org.apache.spark.SparkContext

trait VCFImporter {
  def globAllVcfs(arguments: Array[String], hConf: hadoop.conf.Configuration, forcegz: Boolean = false): Array[String] = {
    val inputs = hConf.globAll(arguments)

    if (inputs.isEmpty)
      fatal("arguments refer to no files")

    inputs.foreach { input =>
      if (!input.endsWith(".vcf")
        && !input.endsWith(".vcf.bgz")) {
        if (input.endsWith(".vcf.gz")) {
          if (!forcegz)
            fatal(".gz cannot be loaded in parallel, use .bgz or -f override")
        } else
          fatal("unknown input file type")
      }
    }
    inputs
  }
}

object ImportVCF extends Command with VCFImporter {
  def name = "importvcf"

  def description = "Load file (.vcf or .vcf.bgz) as the current dataset"

  class Options extends BaseOptions {
    @Args4jOption(name = "-i", aliases = Array("--input"), usage = "Input file (deprecated)")
    var input: Boolean = false

    @Args4jOption(name = "-d", aliases = Array("--no-compress"), usage = "Don't compress in-memory representation")
    var noCompress: Boolean = false

    @Args4jOption(name = "-f", aliases = Array("--force"), usage = "Force load .gz file")
    var force: Boolean = false

    @Args4jOption(name = "--force-bgz", usage = "Force load .gz file using BGzip codec")
    var forceBGz: Boolean = false

    @Args4jOption(name = "--header-file", usage = "File to load header from")
    var headerFile: String = null

    @Args4jOption(name = "-n", aliases = Array("--npartition"), usage = "Number of partitions")
    var nPartitions: Int = 0

    @Args4jOption(name = "--skip-genotypes", usage = "Don't load genotypes")
    var skipGenotypes: Boolean = false

    @Args4jOption(name = "--store-gq", usage = "Store GQ instead of computing from PL")
    var storeGQ: Boolean = false

    @Args4jOption(name = "--pp-as-pl", usage = "Store PP genotype field as Hail PLs [EXPERIMENTAL]")
    var ppAsPL: Boolean = false

    @Args4jOption(name = "--skip-bad-ad", usage = "Set to missing all AD fields with the wrong number of elements")
    var skipBadAD: Boolean = false

    @Argument(usage = "<files...>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()
  }

  def newOptions = new Options

  def supportsMultiallelic = true

  def requiresVDS = false

  def run(state: State, options: Options): State = {
    if (options.input)
      warn("-i deprecated, no longer needed")

    val inputs = globAllVcfs(options.arguments.asScala.toArray, state.hadoopConf, options.force || options.forceBGz)

    val headerFile = if (options.headerFile != null)
      options.headerFile
    else
      inputs.head

    val codecs = state.sc.hadoopConfiguration.get("io.compression.codecs")

    if(options.forceBGz)
      state.sc.hadoopConfiguration.set("io.compression.codecs",
        codecs.replaceAllLiterally("org.apache.hadoop.io.compress.GzipCodec","is.hail.io.compress.BGzipCodecGZ"))

  val vds =  LoadVCF(state.sc,
    headerFile,
    inputs,
    options.storeGQ,
    !options.noCompress,
    if (options.nPartitions != 0)
      Some(options.nPartitions)
    else
      None,
    options.skipGenotypes,
    options.ppAsPL,
    options.skipBadAD)

    state.sc.hadoopConfiguration.set("io.compression.codecs",codecs)

    state.copy(vds = vds)
  }

}
