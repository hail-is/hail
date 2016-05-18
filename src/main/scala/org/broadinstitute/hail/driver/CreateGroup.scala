package org.broadinstitute.hail.driver


import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}
import scala.collection.mutable.ArrayBuffer

class GenotypeStreamCombiner(nSamples: Int, aggregator: (Option[Double], Option[Double]) => Option[Double]) extends Serializable {
  val resultVector = Array.fill[Option[Double]](nSamples)(None)

  def addCount(gs: Iterable[Genotype], ec: EvalContext, f: () => Option[Any], t: (Any) => Double): GenotypeStreamCombiner = {
    require(gs.size == nSamples)
    gs.zipWithIndex.foreach{case (g: Genotype, i: Int) =>
      ec.setAll(g)
      val x = f().map{t}
      resultVector(i) = (resultVector(i) ++ x).reduceOption(_ + _)
    }
    this
  }

  def combineCounts(other: GenotypeStreamCombiner): GenotypeStreamCombiner = {
    other.resultVector.zipWithIndex.foreach{case (g2, i) => resultVector(i) = aggregator(resultVector(i), g2)}
    this
  }

  override def toString: String = {
    resultVector.mkString(",")
  }
}

object CreateGroup extends Command {

  class Options extends BaseOptions {

    @Args4jOption(required = false, name = "-k", aliases = Array("--keys"),
      usage = "comma-separated list of annotations to be used as grouping variable(s) (must be attribute of va)")
    var groupKeys: String = _

    @Args4jOption(required = false, name = "-v", aliases = Array("--variable"),
      usage = "genotype variable to group (must be attribute of g)")
    var groupValues: String = _

    @Args4jOption(required = false, name = "-a", aliases = Array("--aggregator"),
      usage = "function for combining variables across variants in each group [sum, carrier]")
    var aggregator: String = _

  }

  def newOptions = new Options

  def name = "creategroup"

  def description = "create groups for burden tests"

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val vds = state.vds
    val sc = state.sc
    val vas = vds.vaSignature
    val sas = vds.saSignature

    def toDouble(t: BaseType, code: String): Any => Double = t match {
      case TInt => _.asInstanceOf[Int].toDouble
      case TLong => _.asInstanceOf[Long].toDouble
      case TFloat => _.asInstanceOf[Float].toDouble
      case TDouble => _.asInstanceOf[Double]
      case TBoolean => _.asInstanceOf[Boolean].toDouble
      case _ => fatal(s"Genotype variable `$code' must be numeric or Boolean, got $t")
    }

    val symTab = Map("g" -> (0, TGenotype))

    val ec = EvalContext(symTab)

    val (xT, xQ) = Parser.parse(options.groupValues, ec)
    val xToDouble = toDouble(xT, options.groupValues)

    val aggregator = options.aggregator match {
      case "sum" => (a: Option[Double], b: Option[Double]) => (a ++ b).reduceOption(_ + _)
      case "carrier" => (a: Option[Double], b: Option[Double]) => (a ++ b).reduceOption[Double]{case (d1, d2) => math.min(math.min(d1, d2), 1.0)}
      case _ => fatal(s"Option for '-a' ${options.aggregator} not recognized.")
    }
    
    val queriers = options.groupKeys.split(",").map{vds.queryVA(_)._2}

    state.copy(group = vds.rdd.map{case (v, va, gs) => (queriers.map{_(va).get}.toIndexedSeq, gs)}
      .aggregateByKey(new GenotypeStreamCombiner(vds.nSamples, aggregator))(
        (comb, gs) => comb.addCount(gs, ec, xQ, xToDouble), (comb1, comb2) => comb1.combineCounts(comb2))
      .map{case (k, v) => (k, v.resultVector)}
    )
  }
}
