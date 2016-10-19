package org.broadinstitute.hail.driver

import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.methods.Aggregators
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}

object AddKeyTable extends Command {
  class Options extends BaseOptions with TextTableOptions {
    @Args4jOption(required = true, name = "-k", aliases = Array("--key-cond"),
      usage = "Struct with expr defining keys")
    var keyCond: String = _

    @Args4jOption(required = false, name = "-c", aliases = Array("--cond"),
      usage = "Aggregation condition")
    var cond: String = _

    @Args4jOption(required = false, name = "-o", aliases = Array("--output"),
      usage = "output file")
    var outFile: String = _
  }

  def newOptions = new Options

  def name = "addkeytable"

  def description = "Creates new key table with key determined by an expression"

  def supportsMultiallelic = true

  def requiresVDS = true

  override def hidden = true

  def run(state: State, options: Options): State = {

    val vds = state.vds
    val splat = false

//    val cond = options.cond
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

    val (header, parseTypes, f) = Parser.parseNamedArgs(keyCond, ec)

    if (header.isEmpty)
      fatal("this module requires one or more named expr arguments")

//    def buildKeyAggregations(vds: VariantDataset, ec: EvalContext) = {
//      val aggregators = ec.aggregationFunctions.toArray
//      val aggregatorA = ec.a
//
//      if (aggregators.isEmpty)
//        None
//      else {
//
//        val localSamplesBc = vds.sampleIdsBc
//        val localAnnotationsBc = vds.sampleAnnotationsBc
//
//        val nAggregations = aggregators.length
//        val nSamples = vds.nSamples
//        val depth = HailConfiguration.treeAggDepth(vds.nPartitions)
//
//        val baseArray = MultiArray2.fill[Aggregator](nSamples, nAggregations)(null)
//        for (i <- 0 until nSamples; j <- 0 until nAggregations) {
//          baseArray.update(i, j, aggregators(j).copy())
//        }
//      }


    println(header.mkString("\n"))
    println(parseTypes.mkString("\n"))
    println(f().mkString("\n"))

    val foo = vds.rdd.map{case (v, (va, gs)) =>
      ec.set(0, v)
      ec.set(1, va)
      val (header, parseTypes, f) = Parser.parseNamedArgs(keyCond, ec)
      f()
    }

//    println(foo.collect().map(_.mkString(",")).mkString("\n"))



//    val (zVals, seqOp, combOp, resultOp) = Aggregators.makeFunctions(aggregationEC)
//
//    val zvf = () => zVals.indices.map(zVals).toArray
//
//    val results = vds.variantsAndAnnotations.flatMap { case (v, va) => i => (i, (v, va)) }
//    }
//      .aggregateByKey(zvf())(seqOp, combOp)
//      .collectAsMap()

//    println(parseTypes.mkString("\n"))


//    val groups = vds.rdd.flatMap { case (v, (va, gs)) =>
//      val key = qGroupKey(va)
//      val genotypes = gs.map { g => g.nNonRefAlleles.getOrElse(9) } //SKAT-O null value is +9
//      key match {
//        case Some(x) =>
//          if (splat)
//            for (k <- x.asInstanceOf[Iterable[_]]) yield (k, genotypes)
//          else
//            Some((x, genotypes))
//        case None => None
//      }
//    }.groupByKey()



    state
  }
}
