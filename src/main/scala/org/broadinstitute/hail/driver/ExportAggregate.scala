package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils.{MultiArray2}
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}

object ExportAggregate extends Command {

  class Options extends BaseOptions {

    @Args4jOption(required = false, name = "-o", aliases = Array("--output"),
      usage = "path of output file")
    var output: String = _

    @Args4jOption(required = true, name = "-k", aliases = Array("--key-condition"),
      usage = "named expression for which keys to aggregate on (variant and sample)")
    var keyCondition: String = _

    @Args4jOption(required = true, name = "-a", usage = "named expression for item to compute")
    var aggCondition: String = _
  }

  def newOptions = new Options

  def name = "exportaggregate"

  def description = "Aggregate and export samples information grouped by a given variant annnotation"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val sc = vds.sparkContext
    val keyCond = options.keyCondition
    val aggCond = options.aggCondition
    val output = options.output
    val vas = vds.vaSignature
    val sas = vds.saSignature
    val localSamplesBc = vds.sampleIdsBc
    val localAnnotationsBc = vds.sampleAnnotationsBc

    val aggregationEC = EvalContext(Map(
      "v" -> (0, TVariant),
      "va" -> (1, vds.vaSignature),
      "s" -> (2, TSample),
      "sa" -> (3, vds.saSignature),
      "global" -> (4, vds.globalSignature)))

    val ec = EvalContext(Map(
      "v" -> (0, TVariant),
      "va" -> (1, vds.vaSignature),
      "s" -> (2, TSample),
      "sa" -> (3, vds.saSignature),
      "global" -> (4, vds.globalSignature),
      "gs" -> (-1, BaseAggregable(aggregationEC, TGenotype))))

    aggregationEC.set(4, vds.globalAnnotation)
    ec.set(4, vds.globalAnnotation)

    val (aggNames, aggTypes, aggF) = Parser.parseNamedArgs(aggCond, ec)

    if (aggNames.isEmpty)
      fatal("need at least 1 aggregation argument")

    val aggregators = aggregationEC.aggregationFunctions.toArray
    val aggregatorA = aggregationEC.a
    val nAggregations = aggregators.length

    val keyParseResult = Parser.parseNamedArgs(keyCond, ec)

    val sampleGroups = vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      ec.set(2, s)
      ec.set(3, sa)

      keyParseResult._3.apply().toIndexedSeq
    }

    //    val variantGroupEC = EvalContext( Map(
    //      "v" -> (0, TVariant),
    //      "va" -> (1, vds.vaSignature),
    //      "global" -> (2, vds.globalSignature)))
    //    variantGroupEC.set(2,vds.globalSignature)
    //
    //    val variantGroupParseResult = Parser.parseNamedArgs(options.byV ,variantGroupEC)

    val distinctSampleGroupMap = sampleGroups.distinct.zipWithIndex.toMap
    val siToGroupIndex = sampleGroups.map(distinctSampleGroupMap)
    val nSampleGroups = distinctSampleGroupMap.size

    def zero() = {
      val baseArray = MultiArray2.fill[Aggregator](nSampleGroups, nAggregations)(null)
      for (i <- 0 until nSampleGroups; j <- 0 until nAggregations) {
        baseArray.update(i, j, aggregators(j).copy())
      }
      baseArray
    }

    val mapOp :  (Variant, Annotation) => IndexedSeq[Any] =  {case (v, va) =>
      ec.set(0, v)
      ec.set(1, va)
      keyParseResult._3.apply().toIndexedSeq
    }

    val seqOp : (MultiArray2[Aggregator], (Variant, (Annotation, Iterable[Genotype]))) => MultiArray2[Aggregator] = {
      case (arr, (v, (va, gs))) =>
        aggregatorA(0) = v
        aggregatorA(1) = va
        for ((g, i) <- gs.zipWithIndex)
          for (j <- 0 until nAggregations) {
            aggregatorA(2) = localSamplesBc.value(i)
            aggregatorA(3) = localAnnotationsBc.value(i)
            val sampleGroup = siToGroupIndex(i)
            arr(sampleGroup, j).seqOp(g)
          }

        arr
    }

    val combOp : (MultiArray2[Aggregator], MultiArray2[Aggregator]) => MultiArray2[Aggregator] = {
      case (arr1, arr2) =>
        for ((i, j) <- arr1.indices) {
          val a1 = arr1(i, j)
          a1.combOp(arr2(i, j).asInstanceOf[a1.type])
        }
        arr1
    }

    val res = vds.rdd.map { case (v, (va, gs)) => (mapOp(v, va), (v, (va, gs))) }
      .aggregateByKey(zero())(seqOp, combOp)

//    res.map{case (key, agg) => key.mkString(",")}.collect().foreach(println(_))

//
//    def getLine(sampleGroupIndex: Integer, values: MultiArray2[Any], sb:StringBuilder) : String = {
//      for (j <- 0 until nAggregations) {
//        aggregatorA(aggregators(j).idx) = values(sampleGroupIndex, j)
//      }
//
//      aggregationParseResult.foreachBetween { case (t, f) =>
//        sb.append(f().map(TableAnnotationImpex.exportAnnotation(_, t)).getOrElse("NA"))
//      } { sb += '\t' }
//      sb.result()
//    }
//
//      res.map({
//        case (variantGroup, values) =>
//
//          val sb = new StringBuilder()
//          val lines = for ((sampleGroup, i) <- distinctSampleGroupMap.keys.zipWithIndex) yield {
//            sb.clear()
//            sb.append(sampleGroup.map(_.getOrElse("NA").toString).mkString("\t") + "\t")
//            getLine(i,values,sb)
//          }
//          lines.map(variantGroup.map(_.getOrElse("NA").toString).mkString("\t") + "\t" + _).mkString("\n")
//      })
//        .writeTable(options.output,
//          header = Some(variantGroupParseResult.map(_._1).mkString("\t") + "\t" +
//            sampleGroupsParseResult.map(_._1).mkString("\t") + "\t" +
//            aggregationHeader.mkString("\t")))
//

//
    state
  }
}
