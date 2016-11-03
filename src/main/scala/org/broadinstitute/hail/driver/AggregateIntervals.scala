package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.io.annotators.IntervalListAnnotator
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.utils.Interval
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}

import scala.collection.mutable

object AggregateIntervals extends Command {

  class Options extends BaseOptions {

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "path of output file")
    var output: String = _

    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = ".columns file, or comma-separated list of fields/computations")
    var condition: String = _

    @Args4jOption(required = true, name = "-i", usage = "path to interval file")
    var input: String = _

  }

  def newOptions = new Options

  def name = "aggregateintervals"

  def description = "Aggregate and export information over intervals"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val sc = vds.sparkContext
    val cond = options.condition
    val output = options.output
    val vas = vds.vaSignature
    val sas = vds.saSignature

    val aggregationEC = EvalContext(Map(
      "va" -> (0, vds.vaSignature),
      "global" -> (1, vds.globalSignature)))
    val symTab = Map(
      "interval" -> (0, TInterval),
      "global" -> (1, vds.globalSignature),
      "variants" -> (-1, BaseAggregable(aggregationEC, TVariant)))

    val ec = EvalContext(symTab)
    ec.set(1, vds.globalAnnotation)
    aggregationEC.set(1, vds.globalAnnotation)

    val (header, _, f) = Parser.parseExportArgs(cond, ec)

    if (header.isEmpty)
      fatal("this module requires one or more named expr arguments")

    val (zVals, seqOp, combOp, resultOp) = Aggregators.makeFunctions(aggregationEC)

    val zvf = () => zVals.indices.map(zVals).toArray

    val iList = IntervalListAnnotator.read(options.input, sc.hadoopConfiguration)
    val iListBc = sc.broadcast(iList)

    val results = vds.variantsAndAnnotations.flatMap { case (v, va) =>
      iListBc.value.query(v.locus).map { i => (i, (v, va)) }
    }
      .aggregateByKey(zvf())(seqOp, combOp)
      .collectAsMap()

    sc.hadoopConfiguration.writeTextFile(options.output) { out =>
      val sb = new StringBuilder
      sb.append("Contig")
      sb += '\t'
      sb.append("Start")
      sb += '\t'
      sb.append("End")
      header.foreach { col =>
        sb += '\t'
        sb.append(col)
      }
      sb += '\n'

      iList.toIterator
        .foreachBetween { interval =>

          sb.append(interval.start.contig)
          sb += '\t'
          sb.append(interval.start.position)
          sb += '\t'
          sb.append(interval.end.position)
          val res = results.getOrElse(interval, zvf())
          resultOp(res)

          ec.set(0, interval)
          f().foreach { field =>
            sb += '\t'
            sb.append(field)
          }
        }(sb += '\n')

      out.write(sb.result())
    }

    state
  }
}
