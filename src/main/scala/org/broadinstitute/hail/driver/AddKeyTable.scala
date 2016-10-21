package org.broadinstitute.hail.driver

import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods.Aggregators
import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object AddKeyTable extends Command {
  class Options extends BaseOptions with TextTableOptions {
    @Args4jOption(required = true, name = "-k", aliases = Array("--key-cond"),
      usage = "Struct with expr defining keys")
    var keyCond: String = _

    @Args4jOption(required = true, name = "-a", aliases = Array("--agg-cond"),
      usage = "Aggregation condition")
    var aggCond: String = _

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "output file")
    var output: String = _
  }

  def newOptions = new Options

  def name = "addkeytable"

  def description = "Creates new key table with key determined by an expression"

  def supportsMultiallelic = true

  def requiresVDS = true

  override def hidden = true

  def run(state: State, options: Options): State = {

    val vds = state.vds
    val sc = state.sc

    val aggCond = options.aggCond
    val keyCond = options.keyCond

    val aggregationEC = EvalContext(Map(
      "v" -> (0, TVariant),
      "va" -> (1, vds.vaSignature),
      "s" -> (2, TSample),
      "sa" -> (3, vds.saSignature),
      "global" -> (4, vds.globalSignature)))

    val symTab = Map(
      "v" -> (0, TVariant),
      "va" -> (1, vds.vaSignature),
      "s" -> (2, TSample),
      "sa" -> (3, vds.saSignature),
      "global" -> (4, vds.globalSignature),
      "gs" -> (-1, BaseAggregable(aggregationEC, TGenotype)))

    val ec = EvalContext(symTab)
    val a = ec.a

    ec.set(4, vds.globalAnnotation)
    aggregationEC.set(4, vds.globalAnnotation)

    val (keyNames, keyParseTypes, keyF) = Parser.parseNamedArgs(keyCond, ec)
    val (aggNames, aggParseTypes, aggF) = Parser.parseNamedArgs(aggCond, ec)

    if (keyNames.isEmpty)
      fatal("this module requires one or more named expr arguments as keys")
    if (aggNames.isEmpty)
      fatal("this module requires one or more named expr arguments to aggregate by key")

    val (zVals, seqOp, combOp, resultOp) = Aggregators.makeKeyFunctions(aggregationEC)
    val zvf = () => zVals.indices.map(zVals).toArray

    val results = vds.mapPartitionsWithAll{ it =>
      it.map { case (v, va, s, sa, g) =>
        ec.setAll(v, va, s, sa, g)
        val key = keyF().toIndexedSeq
        (key, (v, va, s, sa, g))
        }
    }.aggregateByKey(zvf())(seqOp, combOp).collectAsMap()

    sc.hadoopConfiguration.writeTextFile(options.output) { out =>
      val sb = new StringBuilder
      val headerNames = keyNames ++ aggNames
      headerNames.foreachBetween(k => sb.append(k))(sb += '\t')
      sb += '\n'

      results.foreachBetween { case (key, agg) =>
        key.foreachBetween(k => sb.append(k))(sb += '\t')

        resultOp(agg)

        aggF().foreach { field =>
          sb += '\t'
          sb.append(field)
        }
      }(sb += '\n')

      out.write(sb.result())
    }

    state
  }
}
