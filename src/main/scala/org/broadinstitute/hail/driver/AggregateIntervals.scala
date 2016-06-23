package org.broadinstitute.hail.driver

import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods._
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
      "v" ->(0, TVariant),
      "va" ->(1, vds.vaSignature),
      "global" ->(2, vds.globalSignature)))
    val symTab = Map(
      "interval" ->(0, TStruct("contig" -> TString, "start" -> TInt, "end" -> TInt)),
      "global" ->(1, vds.globalSignature),
      "variants" ->(-1, TAggregable(aggregationEC)))

    val ec = EvalContext(symTab)
    ec.set(1, vds.globalAnnotation)
    aggregationEC.set(2, vds.globalAnnotation)

    val (h, fs) = if (cond.endsWith(".columns"))
      ExportTSV.parseColumnsFile(ec, cond, vds.sparkContext.hadoopConfiguration)
    else
      Parser.parseExportArgs(cond, ec)
    val header = h.get.split("\t")

    if (header.isEmpty)
      fatal("this module requires one or more named expr arguments")

    val (zVals, seqOp, combOp, resultOp) = Aggregators.makeFunctions(aggregationEC)

    val zvf: () => Array[Any] = () => zVals.indices.map(zVals).toArray

    val variantAggregations = Aggregators.buildVariantaggregations(vds, aggregationEC)

    val gis = GenomicIntervalSet.read(options.input, sc.hadoopConfiguration)
    val gisBc = sc.broadcast(gis)

    val results = vds.variantsAndAnnotations.treeAggregate(mutable.Map.empty[GenomicInterval, Array[Any]])({
      case (m, (v, va)) =>
        val intervals = gisBc.value.query(v.contig, v.start)
        intervals.foreach(i => m += (i -> seqOp(m.getOrElse(i, zvf()), (v, va))))
        m
    }, { case (m1, m2) =>
      m2.foreach { case (interval, res) =>
        val m1Results = m1.getOrElse(interval, zVals)
        m1 + (interval -> combOp(res, m1Results))
      }
      m1
    })

    writeTextFile(options.output, sc.hadoopConfiguration) { out =>
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

      gis.intervals
        .toArray
        .sorted
        .iterator
        .foreachBetween { interval =>

          sb.append(interval.contig)
          sb += '\t'
          sb.append(interval.start)
          sb += '\t'
          sb.append(interval.end)
          val res = results.getOrElse(interval, zvf())
          resultOp(res)

          ec.set(0, Annotation(interval.contig, interval.start, interval.end))
          fs.map(_ ()).foreach { r =>
            sb += '\t'
            sb.tsvAppend(r)
          }
        }(() => sb += '\n')

      out.write(sb.result())
    }

    state
  }
}
