package org.broadinstitute.hail.driver

import org.broadinstitute.hail.expr.{Parser, EvalContext, TSample}
import org.broadinstitute.hail.stats.FisherExactTest
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

    val (yT, yQ) = Parser.parse(options.ySA, ec)

    val phenotypes = state.vds.sampleIdsAndAnnotations.map{case (s, sa) =>
      a(0) = s
      a(1) = sa
      (s, yQ())
    }

    val sampleIds = vds.sampleIds
    val nonMissingSamplesAndPheno = phenotypes.filter{ case (s, p) => p.isDefined}.map{case (s, p) => (s, p.getOrElse("None"))}
    val nonMissingSamples = nonMissingSamplesAndPheno.map(_._1)
    val nonMissingPheno = nonMissingSamplesAndPheno.map(_._2)
    val phenotypeCategories = nonMissingSamplesAndPheno.map{case (s, p) => p}.toSet.toArray

    if (phenotypeCategories.length != 2)
      fatal("Can only perform FET on phenotypes with two variables")

    val filteredGroupData = group.map{ case (k, v) =>
      (k, v.zip(sampleIds)
      .filter{case (d, s) => nonMissingSamples.contains(s)}
        .map{case (d, s) => d}.zipWithIndex.map{case (d, i) => (nonMissingPheno(i), d)}.toIndexedSeq)
    }

    val counts = filteredGroupData.map{case (k, v) => (k, v.reduceByKey{case (d1, d2) =>
      (Some(d1._1.getOrElse(0d) + d2._1.getOrElse(0d)), Some(d1._2.getOrElse(0d) + d2._2.getOrElse(0d)))
    })}

    hadoopDelete(options.output, state.hadoopConf, recursive = true)

    val header = s"groupKey\t${phenotypeCategories(0)}_mac\t${phenotypeCategories(0)}_maj\t${phenotypeCategories(1)}_mac\t${phenotypeCategories(1)}_maj\tp"

    counts.mapPartitions { it =>
      val sb = new StringBuilder()
      it.map { case (k, m) =>
        sb.clear()

        val counts = phenotypeCategories.flatMap { p =>
          val (mac, maj) = m(p)
          Array(mac.getOrElse(0d), maj.getOrElse(0d))
        }

        val pval = {
          if (counts(0) + counts(1) == 0d || counts(2) + counts(3) == 0d)
            None
          else
            FisherExactTest(counts).calcPvalue()
        }

        sb.append(k.mkString(","))
        sb.append("\t")
        counts.foreach{d => sb.append(d.toString); sb.append("\t")}
        sb.append(pval) //FIXME: should we add rounding to a certain number of decimal places?
        sb.result()
      }
    }.writeTable(options.output, Some(header))

    state
  }
}
