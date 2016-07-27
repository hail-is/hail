package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.vcf.BufferedLineIterator
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.io.Source

object WriteKudu extends Command {
  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "Output file")
    var output: String = _

    @Args4jOption(required = true, name = "-t", aliases = Array("--table"), usage = "Table name")
    var table: String = _

    @Args4jOption(required = true, name = "-m", aliases = Array("--master"), usage = "Kudu master address")
    var master: String = _

    @Args4jOption(required = true, name = "--vcf-seq-dict", usage = "VCF file to read sequence dictionary from")
    var vcfSeqDict: String = _

    @Args4jOption(required = false, name = "--rows-per-partition", usage = "Number of rows in each Kudu partition (tablet)")
    var rowsPerPartition: Int = 30000000 // give <120 tablets on test cluster (see --max_create_tablets_per_ts)

    @Args4jOption(required = false, name = "--sample-group", usage = "Arbitrary unique identifier for the group of samples being imported.")
    var sampleGroup: String = System.currentTimeMillis.toString

    @Args4jOption(required = false, name = "--compress", usage = "compress genotype streams using LZ4")
    var compress: Boolean = true

    @Args4jOption(required = false, name = "--drop", usage = "Drop table first")
    var drop: Boolean = false
  }
  def newOptions = new Options

  def name = "writekudu"
  def description = "Write current dataset to Kudu"
  override def supportsMultiallelic = false
  override def requiresVDS = true

  def run(state: State, options: Options): State = {
    hadoopDelete(options.output, state.hadoopConf, true)
    state.vds.writeKudu(state.sqlContext, options.output, options.table,
      options.master, options.vcfSeqDict, options.rowsPerPartition, options.sampleGroup,
      compress = options.compress, drop = options.drop)
    state
  }
}
