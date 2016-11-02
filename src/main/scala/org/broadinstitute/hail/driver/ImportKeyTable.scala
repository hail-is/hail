package org.broadinstitute.hail.driver

import org.broadinstitute.hail.expr.{Parser, TStruct}
import org.broadinstitute.hail.keytable.KeyTable
import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Argument, Option => Args4jOption}

import scala.collection.JavaConverters._

object ImportKeyTable extends Command {

  class Options extends BaseOptions with TextTableOptions {
    @Argument(usage = "<files...>")
    var arguments: java.util.ArrayList[String] = new java.util.ArrayList[String]()

    @Args4jOption(required = true, name = "-n", aliases = Array("--name"),
      usage = "name of key table")
    var name: String = _

    @Args4jOption(required = true, name = "-k", aliases = Array("--key-names"),
      usage = "comma-separated list of columns to be considered as keys")
    var keyNames: String = _

    @Args4jOption(name = "--npartition", usage = "Number of partitions")
    var nPartitions: java.lang.Integer = _
  }

  def newOptions = new Options

  def name = "importkeytable"

  def description = "import key table from tsv"

  def supportsMultiallelic = true

  def requiresVDS = false

  override def hidden = true

  def run(state: State, options: Options): State = {
    val files = state.hadoopConf.globAll(options.arguments.asScala)
    if (files.isEmpty)
      fatal("Arguments referred to no files")

    val keyNames = Parser.parseIdentifierList(options.keyNames)

    val (struct, rdd) =
      if (options.nPartitions != null) {
        if (options.nPartitions < 1)
          fatal("requested number of partitions in -n/--npartitions must be positive")
        TextTableReader.read(state.sc)(files, options.config, options.nPartitions)
      } else
        TextTableReader.read(state.sc)(files, options.config)

    val keyNamesValid = keyNames.forall{ k =>
      val res = struct.selfField(k).isDefined
      if (!res)
        println(s"Key `$k' is not present in input table")
      res
    }
    if (!keyNamesValid)
      fatal("Invalid key names given")

    state.copy(ktEnv = state.ktEnv + (options.name -> KeyTable(rdd.map(_.value), struct, keyNames)))
  }
}

