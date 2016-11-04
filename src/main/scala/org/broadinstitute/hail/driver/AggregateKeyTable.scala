package org.broadinstitute.hail.driver

import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.keytable.KeyTable
import org.broadinstitute.hail.methods.Aggregators
import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object AggregateKeyTable extends Command {

  class Options extends BaseOptions {

    @Args4jOption(required = false, name = "-d", aliases = Array("--dest"),
      usage = "name of joined key-table")
    var dest: String = _

    @Args4jOption(required = true, name = "-n", aliases = Array("--name"),
      usage = "name of key-table to aggregate")
    var name: String = _

    @Args4jOption(required = false, name = "-k", aliases = Array("--key-cond"),
      usage = "Named key condition")
    var keyCond: String = _

    @Args4jOption(required = false, name = "-a", aliases = Array("--agg-cond"),
      usage = "Named aggregation condition")
    var aggCond: String = "left"
  }

  def newOptions = new Options

  def name = "aggregatekeytable"

  def description = "Aggregate over fields of key-table to produce new key table"

  def supportsMultiallelic = true

  def requiresVDS = false

  override def hidden = true

  def run(state: State, options: Options): State = {
    val ktEnv = state.ktEnv
    val name = options.name
    val dest = if (options.dest != null) options.dest else name

    val aggCond = options.aggCond
    val keyCond = options.keyCond

    val kt = ktEnv.get(name) match {
      case Some(x) => x
      case None => fatal("no such key table $name in environment")
    }

    if (ktEnv.contains(dest))
      warn("destination name already exists -- overwriting previous key-table")

    val ec = EvalContext(kt.fields.map(fd => (fd.name, fd.`type`)): _*)

    val (keyNameParseTypes, keyF) =
      if (keyCond != null)
        Parser.parseAnnotationArgs(keyCond, ec, None)
      else
        (Array.empty[(List[String], Type)], Array.empty[() => Any])

    val (aggNameParseTypes, aggF) =
      if (aggCond != null)
        Parser.parseAnnotationArgs(aggCond, ec, None)
      else
        (Array.empty[(List[String], Type)], Array.empty[() => Any])

    val keyNames = keyNameParseTypes.map(_._1.head)
    val aggNames = aggNameParseTypes.map(_._1.head)

    val keySignature = TStruct(keyNameParseTypes.map{ case (n, t) => (n.head, t) }: _*)
    val valueSignature = TStruct(aggNameParseTypes.map{ case (n, t) => (n.head, t) }: _*)

    val nKeys = kt.nKeys
    val nValues = kt.nValues

//    val (zVals, _, combOp, resultOp) = Aggregators.makeFunctions(ec.copy())
//
//    val seqOp = (array: Array[Aggregator], b: (Any, Any, Any)) => {
//      val (k, v, aggT) = b
//      KeyTable.setEvalContext(ec, k, v, nKeys, nValues)
//      for (i <- array.indices) {
//        array(i).seqOp(aggT)
//      }
//      array
//    }
//
//    kt.mapAnnotations { (k, v) =>
//      KeyTable.setEvalContext(ec, k, v, nKeys, nValues)
//      val key = Annotation.fromSeq(keyF.map(_ ()))
//      (key, (k, v))
//    }.aggregateByKey(zVals)(seqOp, combOp) // FIXME: need to aggregate .aggregateByKey()

    val ktAgg = kt // FIXME: place holder for now
    state.copy(ktEnv = state.ktEnv + (dest -> ktAgg))
  }
}

