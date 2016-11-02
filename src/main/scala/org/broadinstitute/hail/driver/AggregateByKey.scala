package org.broadinstitute.hail.driver

import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.keytable.KeyTable
import org.broadinstitute.hail.methods.Aggregators
import org.kohsuke.args4j.{Option => Args4jOption}

object AggregateByKey extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-k", aliases = Array("--key-cond"),
      usage = "Named key condition", metaVar = "EXPR")
    var keyCond: String = _

    @Args4jOption(required = true, name = "-a", aliases = Array("--agg-cond"),
      usage = "Named aggregation condition", metaVar = "EXPR")
    var aggCond: String = _

    @Args4jOption(required = true, name = "-n", aliases = Array("--name"),
      usage = "Name of new key table")
    var name: String = _
  }

  def newOptions = new Options

  def name = "aggregatebykey"

  def description = "Creates a new key table with key(s) determined by named expressions and additional columns determined by named aggregator expressions"

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

    val (testPT, testF) = Parser.parseAnnotationArgs(aggCond, ec)

    println(testPT.mkString("\n"))


    val signature = TStruct((keyNames ++ aggNames).zip(keyParseTypes ++ aggParseTypes): _*)

    val (zVals, _, combOp, resultOp) = Aggregators.makeFunctions(aggregationEC)
    val zvf = () => zVals.indices.map(zVals).toArray

    val seqOp = (array: Array[Aggregator], b: (Any, Any, Any, Any, Any)) => {
      val (v, va, s, sa, aggT) = b
      ec.set(0, v)
      ec.set(1, va)
      ec.set(2, s)
      ec.set(3, sa)
      for (i <- array.indices) {
        array(i).seqOp(aggT)
      }
      array
    }

    val kt = KeyTable(vds.mapPartitionsWithAll { it =>
      it.map { case (v, va, s, sa, g) =>
        ec.setAll(v, va, s, sa, g)
        val key = keyF(): IndexedSeq[String]
        (key, (v, va, s, sa, g))
      }
    }.aggregateByKey(zvf())(seqOp, combOp)
      .map { case (k, agg) =>
        resultOp(agg)
        Annotation.fromSeq(k ++ aggF())
      }, signature, keyNames)

    state.copy(ktEnv = state.ktEnv + (options.name -> kt))
  }
}
