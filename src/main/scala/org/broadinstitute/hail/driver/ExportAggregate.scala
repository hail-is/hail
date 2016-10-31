package org.broadinstitute.hail.driver

import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils.richUtils.RichRDD
import org.broadinstitute.hail.utils.{MultiArray2}
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}

object ExportAggregate extends Command {

  class Options extends BaseOptions {

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"),
      usage = "path of output file")
    var output: String = _

    @Args4jOption(required = true, name = "-c", aliases = Array("--condition"),
      usage = ".columns file, or comma-separated list of fields/computations")
    var condition: String = _

    @Args4jOption(required = true, name = "--byV", usage = "fileds/expression to aggregate variants on.")
    var byV: String = _

    @Args4jOption(required = false, name = "--byS", usage = "expression to aggregate samples")
    var byS: String = "Sample=s.id"

    @Args4jOption(required = false, name = "--as-matrix", usage = "When using this option, only a single condition can " +
      "be passed. If set, the output is a matrix of variants x samples with each cell containing the value of the condition.")
    var asMatrix: Boolean = false

  }

  def newOptions = new Options

  def name = "exportaggregate"

  def description = "Aggregate and export samples information grouped by a given variant annnotation"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val sc = vds.sparkContext
    val cond = options.condition
    val output = options.output
    val vas = vds.vaSignature
    val sas = vds.saSignature
    val asMatrix = options.asMatrix
    val localSamplesBc = vds.sampleIdsBc
    val localAnnotationsBc = vds.sampleAnnotationsBc

    val aggregationEC = EvalContext(Map(
      "v" -> (0, TVariant),
      "va" -> (1, vds.vaSignature),
      "s" -> (2, TSample),
      "sa" -> (3, vds.saSignature),
      "global" -> (4, vds.globalSignature)))
    aggregationEC.set(4, vds.globalAnnotation)

    val ec = EvalContext(Map(
      "global" -> (0, vds.globalSignature),
      "gs" -> (-1, BaseAggregable(aggregationEC, TGenotype))))
    ec.set(0, vds.globalAnnotation)


    val (aggregationHeader, aggregationParseResult) = if (cond.endsWith(".columns")) {
      Parser.parseColumnsFile(ec, cond, vds.sparkContext.hadoopConfiguration)
    } else {
      val ret = Parser.parseNamedArgs(cond, ec)
      (ret.map(_._1), ret.map(x => (x._2, x._3)))
    }

    if (aggregationHeader.isEmpty)
      fatal("this module requires one or more named expr arguments")

    if(asMatrix && aggregationHeader.length > 1)
      fatal("Only a single condition can be evaluated when using --as-matrix.")

    val aggregators = aggregationEC.aggregationFunctions.toArray
    val aggregatorA = aggregationEC.a
    val nAggregations = aggregators.length

    val variantGroupEC = EvalContext( Map(
    "v" -> (0, TVariant),
    "va" -> (1, vds.vaSignature),
    "global" -> (2, vds.globalSignature)))
    variantGroupEC.set(2,vds.globalSignature)

    val variantGroupParseResult = Parser.parseNamedArgs(options.byV ,variantGroupEC)

    val sampleGroupEC = EvalContext(Map(
    "s" -> (0 -> TSample),
    "sa" -> (1 -> vds.saSignature),
    "global" -> (2, vds.globalSignature)))
    sampleGroupEC.set(2,vds.globalSignature)

    val sampleGroupsParseResult = Parser.parseNamedArgs(options.byS, sampleGroupEC)
    val sampleGroups = vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      sampleGroupEC.set(0, s)
      sampleGroupEC.set(1, sa)

      sampleGroupsParseResult.map { case (_, _, f) => f()}.toIndexedSeq
    }
    val distinctSampleGroupMap = sampleGroups.distinct.zipWithIndex.toMap
    val siToGroupIndex = sampleGroups.map(distinctSampleGroupMap)
    val nSampleGroups = distinctSampleGroupMap.size

    def zero() = {
      val baseArray = MultiArray2.fill[Any](nSampleGroups, nAggregations)(null)
      for (i <- 0 until nSampleGroups; j <- 0 until nAggregations) {
        baseArray.update(i, j, aggregators(j).zero)
      }
      baseArray
    }

    val mapOp :  (Variant, Annotation) => IndexedSeq[Option[Any]] =  {case (v,va) =>
      variantGroupEC.set(0, v)
      variantGroupEC.set(1, va)
      variantGroupParseResult.map(_._3()).toIndexedSeq
    }

    val seqOp : (MultiArray2[Any], (Variant, (Annotation, Iterable[Genotype]))) => MultiArray2[Any] = {
      case (arr, (v, (va, gs))) =>
        aggregatorA(0) = v
        aggregatorA(1) = va
        for ((g, i) <- gs.zipWithIndex)
          for (j <- 0 until nAggregations) {
            aggregatorA(2) = localSamplesBc.value(i)
            aggregatorA(3) = localAnnotationsBc.value(i)
            val sampleGroup = siToGroupIndex(i)
            arr.update(sampleGroup, j, aggregators(j).seqOp(g, arr(sampleGroup, j)))
          }

        arr
    }
    val combOp : (MultiArray2[Any],MultiArray2[Any]) => MultiArray2[Any] = {
      case (arr1, arr2) =>
        for (i <- 0 until arr1.n1; j <- 0 until arr1.n2)
          arr1.update(i, j, aggregators(j).combOp(arr1(i, j), arr2(i, j)))
        arr1
    }

    val res = vds.rdd.map { case (v, (va, gs)) => (mapOp(v, va), (v, (va, gs))) }
      .aggregateByKey(zero())(seqOp, combOp)


    def getLine(sampleGroupIndex: Integer, values: MultiArray2[Any], sb:StringBuilder) : String = {
      for (j <- 0 until nAggregations) {
        aggregatorA(aggregators(j).idx) = values(sampleGroupIndex, j)
      }

      aggregationParseResult.foreachBetween { case (t, f) =>
        sb.append(f().map(TableAnnotationImpex.exportAnnotation(_, t)).getOrElse("NA"))
      } { sb += '\t' }
      sb.result()
    }

    if (asMatrix) {
      res.map({
        case (variantGroup, values) =>

          val sb = new StringBuilder()
          val lines = for (i <- 0 until nSampleGroups) yield {
            sb.clear()
            getLine(i,values,sb)
          }
          variantGroup.map(_.getOrElse("NA").toString).mkString("_") + "\t" + lines.mkString("\t")
      })
        .writeTable(options.output,
          header = Some(variantGroupParseResult.map(_._1).mkString("_") + "\t" +
            distinctSampleGroupMap.keys.map(_.map(_.getOrElse("NA").toString).mkString("_")).mkString("\t")))
    } else {
      res.map({
        case (variantGroup, values) =>

          val sb = new StringBuilder()
          val lines = for ((sampleGroup, i) <- distinctSampleGroupMap.keys.zipWithIndex) yield {
            sb.clear()
            sb.append(sampleGroup.map(_.getOrElse("NA").toString).mkString("\t") + "\t")
            getLine(i,values,sb)
          }
          lines.map(variantGroup.map(_.getOrElse("NA").toString).mkString("\t") + "\t" + _).mkString("\n")
      })
        .writeTable(options.output,
          header = Some(variantGroupParseResult.map(_._1).mkString("\t") + "\t" +
            sampleGroupsParseResult.map(_._1).mkString("\t") + "\t" +
            aggregationHeader.mkString("\t")))
    }

    state
  }
}
