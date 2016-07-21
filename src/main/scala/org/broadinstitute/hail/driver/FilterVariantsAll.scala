package org.broadinstitute.hail.driver

import org.apache.spark.rdd.OrderedRDD
import org.broadinstitute.hail.variant._
import org.broadinstitute.hail.annotations._

object FilterVariantsAll extends Command {

  class Options extends BaseOptions

  def newOptions = new Options

  def name = "filtervariants all"

  def description = "Discard all variants in the current dataset"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    state.copy(vds = state.vds.copy(rdd = OrderedRDD.empty[Locus, Variant, (Annotation, Iterable[Genotype])](state.sc, _.locus)))
  }
}
