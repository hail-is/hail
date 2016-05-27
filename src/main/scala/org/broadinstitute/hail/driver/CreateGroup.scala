package org.broadinstitute.hail.driver


import org.broadinstitute.hail.Utils._
import org.broadinstitute.hail.methods._
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.variant._
import org.kohsuke.args4j.{Option => Args4jOption}
import scala.collection.mutable.ArrayBuffer

class GenotypeStreamCombiner[T](nSamples: Int, aggregator: (Option[T], Option[T]) => Option[T]) extends Serializable {
  val resultVector = Array.fill[Option[T]](nSamples)(None)

  def addCount(gs: Iterable[Genotype], ec: EvalContext, f: () => Option[Any], t: (Any) => T): GenotypeStreamCombiner[T] = {
    require(gs.size == nSamples)

    gs.zipWithIndex.foreach{case (g: Genotype, i: Int) =>
      ec.setAll(g)
      resultVector(i) = aggregator(resultVector(i), f().map(t))
    }
    this
  }

  def combineCounts(other: GenotypeStreamCombiner[T]): GenotypeStreamCombiner[T] = {
    other.resultVector.zipWithIndex.foreach{case (g2, i) => resultVector(i) = aggregator(resultVector(i), g2)}
    this
  }
}

object CreateGroup extends Command {

  class Options extends BaseOptions {

    @Args4jOption(required = false, name = "-k", aliases = Array("--groupkeys"),
      usage = "comma-separated list of annotations to be used as grouping variable(s) (must be attribute of va)")
    var groupKeys: String = _

    @Args4jOption(required = false, name = "-v", aliases = Array("--groupvalues"),
      usage = "genotype variable to group (must be attribute of g)")
    var groupValues: String = _

    @Args4jOption(required = false, name = "-a", aliases = Array("--aggregator"),
      usage = "function for combining variables across variants in each group [sum, carrier]")
    var aggregator: String = _

  }

  def newOptions = new Options

  def name = "creategroup"

  def description = "create groups for burden tests"

  def requiresVDS = true

  override def supportsMultiallelic = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

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
      case "carrier" => (a: Option[Double], b: Option[Double]) => (a ++ b).reduceOption[Double]{case (d1, d2) => math.min(math.max(d1, d2), 1.0)}
      case _ => fatal(s"Option for '-a' ${options.aggregator} not recognized.")
    }

    val queriers = options.groupKeys.split(",").map{vds.queryVA(_)._2}

    state.copy(group = vds.rdd.map{case (v, va, gs) => (queriers.map{_(va).get}.toIndexedSeq, gs)}
      .aggregateByKey(new GenotypeStreamCombiner[Double](vds.nSamples, aggregator))(
        (comb, gs) => comb.addCount(gs, ec, xQ, xToDouble), (comb1, comb2) => comb1.combineCounts(comb2))
      .map{case (k, v) => (k, v.resultVector)}
    )
  }
}
