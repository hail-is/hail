package is.hail.driver

import is.hail.utils._
import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.io.annotators.IntervalListAnnotator
import is.hail.methods._
import is.hail.utils.Interval
import is.hail.variant._
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
    val localGlobalAnnotation = vds.globalAnnotation

    val aggregationST = Map(
      "global" -> (0, vds.globalSignature),
      "interval" -> (1, TInterval),
      "v" -> (2, TVariant),
      "va" -> (3, vds.vaSignature))
    val symTab = Map(
      "global" -> (0, vds.globalSignature),
      "interval" -> (1, TInterval),
      "variants" -> (2, TAggregable(TVariant, aggregationST)))

    val ec = EvalContext(symTab)
    ec.set(1, vds.globalAnnotation)

    val (names, _, f) = Parser.parseExportExprs(cond, ec)

    if (names.isEmpty)
      fatal("this module requires one or more named expr arguments")

    val (zVals, seqOp, combOp, resultOp) = Aggregators.makeFunctions[(Interval[Locus], Variant, Annotation)](ec, { case (ec, (i, v, va)) =>
        ec.setAll(localGlobalAnnotation, i, v, va)
    })

    val iList = IntervalListAnnotator.read(options.input, sc.hadoopConfiguration)
    val iListBc = sc.broadcast(iList)

    val results = vds.variantsAndAnnotations.flatMap { case (v, va) =>
      iListBc.value.query(v.locus).map { i => (i, (i, v, va)) }
    }
      .aggregateByKey(zVals)(seqOp, combOp)
      .collectAsMap()

    sc.hadoopConfiguration.writeTextFile(options.output) { out =>
      val sb = new StringBuilder
      sb.append("Contig")
      sb += '\t'
      sb.append("Start")
      sb += '\t'
      sb.append("End")
      names.foreach { col =>
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
          val res = results.getOrElse(interval, zVals)
          resultOp(res)

          ec.setAll(localGlobalAnnotation, interval)
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
