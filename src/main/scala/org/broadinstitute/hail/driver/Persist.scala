package org.broadinstitute.hail.driver

import org.apache.spark.storage.StorageLevel
import org.broadinstitute.hail.Utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object Persist extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-s", aliases = Array("--storage-level"), usage = "Storage level, one of: NONE, DISK_ONLY, DISK_ONLY_2, MEMORY_ONLY, MEMORY_ONLY_2, MEMORY_ONLY_SER, MEMORY_ONLY_SER_2, MEMORY_AND_DISK, MEMORY_AND_DISK_2, MEMORY_AND_DISK_SER, MEMORY_AND_DISK_SER_2, OFF_HEAP")
    var level: String = _

  }

  def newOptions = new Options

  def name = "persist"

  def description = "Persist the current dataset"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val level = try {
      StorageLevel.fromString(options.level)
    } catch {
      case e: IllegalArgumentException =>
      fatal(s"unknown StorageLevel `${options.level}'")
    }

    state.copy(
      vds = vds.copy(rdd = vds.rdd.persist(level)))
  }
}
