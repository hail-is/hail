package is.hail.driver

import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.methods._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.language.postfixOps

object TDTCommand extends Command {

  def name = "tdt"

  def description = "Test variants for association using the transmission disequilibrium test"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-f", aliases = Array("--fam"), usage = ".fam file")
    var famFilename: String = _

    @Args4jOption(required = false, name = "-r", aliases = Array("--root"), usage = "Annotation root, starting in `va'")
    var root: String = "va.tdt"
  }

  def newOptions = new Options

  def supportsMultiallelic = false

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val ped = Pedigree.read(options.famFilename, state.hadoopConf, state.vds.sampleIds)
    state.copy(vds = TDT(state.vds, ped.completeTrios,
      Parser.parseAnnotationRoot(options.root, Annotation.VARIANT_HEAD)))
  }
}
