package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._

object SparkInfo extends Command {

  class Options extends BaseOptions

  def newOptions = new Options

  def supportsMultiallelic = true

  def requiresVDS = true

  def name = "sparkinfo"

  def description = "Displays the number of partitions and persistence level of the current vds"

  def run(state: State, options: Options): State = {
    info(s"${state.vds.nPartitions} partitions, ${state.vds.rdd.getStorageLevel.toReadableString()}")

    state
  }
}
