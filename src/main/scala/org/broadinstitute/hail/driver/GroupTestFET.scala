package org.broadinstitute.hail.driver

import org.broadinstitute.hail.expr.{Parser, EvalContext, TSample}
import org.kohsuke.args4j.{Option => Args4jOption}
import org.broadinstitute.hail.Utils._

object GroupTestFET extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-y", aliases = Array("--y"), usage = "Response sample annotation")
    var ySA: String = _

    @Args4jOption(required = true, name = "-o", aliases = Array("--output"), usage = "path of output tsv")
    var output: String = _
  }

  def newOptions = new Options

  def name = "grouptest fisher"

  def description = "Run Fisher's exact test on previously defined groups"

  def supportsMultiallelic = true

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val group = state.group

    if (group == null)
      fatal("No group has been created. Use the `creategroup` command.")

    val symTab = Map(
      "s" -> (0, TSample),
      "sa" -> (1, vds.saSignature))

    val ec = EvalContext(symTab)
    val a = ec.a

    //val sampleIds = vds.sampleIdsAndAnnotations.map(_._1)

    val (yT, yQ) = Parser.parse(options.ySA, ec)

    val phenotypes = state.vds.sampleIdsAndAnnotations.map{case (s, sa) =>
      a(0) = s
      a(1) = sa
      (s, yQ())
    }

    val sampleIds = vds.sampleIds
    val nonMissingSamplesAndPheno = phenotypes.filter{ case (s, p) => p.isDefined}
    val nonMissingSamples = nonMissingSamplesAndPheno.map(_._1)
    val nonMissingPheno = nonMissingSamplesAndPheno.map(_._2)
    val numPhenotypeCategories = nonMissingSamplesAndPheno.map{case (s, p) => p}.toSet.size

    val filteredGroupData = group.map{ case (k, v) =>
      (k, v.zip(sampleIds)
      .filter{case (d, s) => nonMissingSamples.contains(s)}
        .map{case (d, s) => d}.zipWithIndex.map{case (d, i) => (nonMissingPheno(i), d)}.toIndexedSeq)
    }

    //filteredGroupData.collect().foreach{case (k, v) => println(k.mkString(",") + " " + v.map{case (p, d) => p.toString + ":" + d}.mkString(","))}
    if (numPhenotypeCategories != 2)
      fatal("Can only perform FET on phenotypes with only two variables")

    val counts = filteredGroupData.map{case (k, v) => (k, v.reduceByKey{case (d1, d2) => Some(d1.getOrElse(0d) + d2.getOrElse(0d))})}

    println(counts.collect().foreach{case (k, v) => println(k + ":" + v.map{case (p, d) => p.toString + "," + d.toString})})
//    val yToDouble = toDouble(yT, yName)
//    val ySA = sampleIdsAndAnnotations.map { case (s, sa) =>
//
//      yQ().map(yToDouble)
//    }
//
//    val sampleIds = vds.sampleIds
//
//    val inputData = group.map{ case (k, v) =>
//      (k, v.zipWithIndex.filter{ case (d, i) =>
//        completeSampleSet(sampleIds(i))
//      }.map{case (d, i) => (y(i), d)}.groupBy{case (p, d)}}
//
//
//    hadoopDelete(options.output, state.hadoopConf, recursive = true)
//
//    val header = "groupKey\tnMissing\tbeta\tse\tt\tp"
//
//    linreg.rdd
//      .mapPartitions { it =>
//        val sb = new StringBuilder()
//        it.map { case (k, v) =>
//          sb.clear()
//          sb.append(k.mkString(","))
//          sb.append("\t")
//          sb.append(v.getOrElse("NA\tNA\tNA\tNA\tNA\tNA\n").toString)
//          sb.result()
//        }
//      }.writeTable(options.output, Some(header))

    state
  }
}
