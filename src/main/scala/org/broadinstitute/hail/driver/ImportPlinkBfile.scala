package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.io.PlinkLoader
import org.kohsuke.args4j.{Option => Args4jOption}

object ImportPlinkBfile extends Command {
  def name = "importplink"

  def description = "Load PLINK binary file (.bed, .bim, .fam) as the current dataset"

  class Options extends BaseOptions {
    @Args4jOption(name = "--bfile", usage = "Plink Binary file root name", forbids = Array("--bed","--bim","--fam"))
    var bfile: String = _

    @Args4jOption(name = "--bed", usage = "Plink .bed file", forbids = Array("--bfile"), depends = Array("--bim","--fam"))
    var bed: String = _

    @Args4jOption(name = "--bim", usage = "Plink .bim file", forbids = Array("--bfile"), depends = Array("--bed","--fam"))
    var bim: String = _

    @Args4jOption(name = "--fam", usage = "Plink .fam file", forbids = Array("--bfile"), depends = Array("--bim","--bed"))
    var fam: String = _
  }

  def newOptions = new Options

  def run(state: State, options: Options): State = {
    if (options.bfile == null && (options.bed == null || options.bim == null || options.fam == null))
      fatal("Invalid input...")
    if (options.bfile != null) {
      state.copy(vds = PlinkLoader(options.bfile, state.sc))
    }
    else {
      state.copy(vds = PlinkLoader(options.bed, options.bim, options.fam, state.sc))
    }
  }
}