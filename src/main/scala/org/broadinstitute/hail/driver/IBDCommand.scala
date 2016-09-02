package org.broadinstitute.hail.driver

import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr.{EvalContext, Parser, TDouble, TVariant}
import org.broadinstitute.hail.methods.IBD
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.{Variant, VariantDataset}
import org.kohsuke.args4j.{Option => Args4jOption}

object IBDCommand extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-m", aliases = Array("--minor-allele-frequency"),
      usage = "An expression for the minor allele frequency of the current variant, `v', given the variant annotations `va'")
    var computeMafExpr: String = _

    @Args4jOption(required = false, name = "--unbounded",
      usage = "Allows the estimations for Z0, Z1, Z2, and PI_HAT to take on biologically-nonsense values (e.g. outside of [0,1]).")
    var unbounded: Boolean = false

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "Output TSV file for the IBD matrix")
    var outputFile: String = _

  }

  override def newOptions = new Options

  override def name: String = "ibd"

  override def description: String = "Compute a matrix of identity-by-descent estimations for each pair of samples"

  override def supportsMultiallelic: Boolean = false

  override def requiresVDS: Boolean = true

  def generateComputeMaf(vds: VariantDataset, computeMafExpr: String): (Variant, Annotation) => Double = {
    val mafSymbolTable = Map("v" -> (0, TVariant), "va" -> (1, vds.vaSignature))
    val mafEc = EvalContext(mafSymbolTable)
    val computeMafThunk = Parser.parse[Double](computeMafExpr, mafEc, TDouble)

    { (v: Variant, va: Annotation) =>
      mafEc.setAll(v, va)
      val maf = computeMafThunk()
        .getOrElse(fatal(s"The minor allele frequency expression evaluated to NA on variant $v."))

      if (maf < 0.0 || maf > 1.0)
        fatal(s"The minor allele frequency expression for $v evaluated to $maf which is not in [0,1].")

      maf
    }
  }

  override protected def run(state: State, options: Options): State = {
    val computeMaf = Option(options.computeMafExpr).map(generateComputeMaf(state.vds, _))

    val ibdMap = IBD(state.vds, computeMaf, !options.unbounded)

    ibdMap.map { case ((i, j), ibd) =>
        s"$i\t$j\t${ibd.ibd.Z0}\t${ibd.ibd.Z1}\t${ibd.ibd.Z2}\t${ibd.ibd.PI_HAT}"
      }
      .saveAsTextFile(options.outputFile)

    state
  }
}
