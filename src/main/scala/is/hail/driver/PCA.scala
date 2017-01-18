package is.hail.driver

import is.hail.methods.SamplePCA
import is.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object PCA extends Command {
  def name = "pca"

  def description = "Run principle component analysis on the matrix of genotypes"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-s", aliases = Array("--scores"),
      usage = "Sample annotation path for scores (period-delimited path starting in `sa')")
    var sRoot: String = _

    @Args4jOption(required = false, name = "-k", aliases = Array("--components"),
      usage = "Number of principal components")
    var k: Int = 10

    @Args4jOption(required = false, name = "-l", aliases = Array("--loadings"),
      usage = "Variant annotation path for site loadings (period-delimited path starting in `va')")
    var lRoot: String = _

    @Args4jOption(required = false, name = "-e", aliases = Array("--eigenvalues"),
      usage = "Global annotation path for eigenvalues (period-delimited path starting in `global'")
    var eRoot: String = _

    @Args4jOption(required = false, name = "-a", aliases = Array("--arrays"),
      usage = "Store score and loading results as arrays, rather than structs")
    var arrays: Boolean = false

  }

  def newOptions = new Options

  def supportsMultiallelic = false

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    var vds = state.vds

    val asArray = options.arrays

    val k = options.k
    if (k < 1)
      fatal(
        s"""requested invalid number of components: $k
            |  Expect componenents >= 1""".stripMargin)

    val pcSchema = SamplePCA.pcSchema(asArray, k)

    val (scores, loadings, eigenvalues) =
      SamplePCA(vds, options.k, Option(options.lRoot).isDefined, Option(options.eRoot).isDefined, asArray)

    vds = vds.annotateSamples(scores, pcSchema, options.sRoot)

    loadings.foreach { rdd =>
      vds = vds.annotateVariants(rdd.orderedRepartitionBy(vds.rdd.orderedPartitioner), pcSchema, options.lRoot)
    }

    eigenvalues.foreach { eig =>
      vds = vds.annotateGlobal(eig, pcSchema, options.eRoot)
    }

    state.copy(vds = vds)
  }
}
