package org.broadinstitute.hail.methods

import org.apache.spark.util.StatCounter
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.driver.HailConfiguration
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.stats.{HistogramCombiner, InfoScoreCombiner}
import org.broadinstitute.hail.variant._

import scala.collection.mutable.ArrayBuffer
import scala.util.parsing.input.Position

object Aggregators {

  def buildVariantAggregations(vds: VariantDataset, ec: EvalContext): Option[(Variant, Annotation, Iterable[Genotype]) => Unit] = {
    val aggregators = ec.aggregationFunctions.toArray
    val aggregatorA = ec.a

    if (aggregators.nonEmpty) {

      val localSamplesBc = vds.sampleIdsBc
      val localAnnotationsBc = vds.sampleAnnotationsBc

      val f = (v: Variant, va: Annotation, gs: Iterable[Genotype]) => {
        val baseArray = aggregators.map(_.copy())
        aggregatorA(0) = v
        aggregatorA(1) = va
        (gs, localSamplesBc.value, localAnnotationsBc.value).zipped
          .foreach {
            case (g, s, sa) =>
              aggregatorA(2) = s
              aggregatorA(3) = sa
              baseArray.foreach {
                _.seqOp(g)
              }
          }

        baseArray.foreach { agg => aggregatorA(agg.idx) = agg.result }
      }
      Some(f)
    } else None
  }

  def buildSampleAggregations(vds: VariantDataset, ec: EvalContext): Option[(String) => Unit] = {
    val aggregators = ec.aggregationFunctions.toArray
    val aggregatorA = ec.a

    if (aggregators.isEmpty)
      None
    else {

      val localSamplesBc = vds.sampleIdsBc
      val localAnnotationsBc = vds.sampleAnnotationsBc

      val nAggregations = aggregators.length
      val nSamples = vds.nSamples
      val depth = HailConfiguration.treeAggDepth(vds.nPartitions)

      val baseArray = MultiArray2.fill[Aggregator](nSamples, nAggregations)(null)
      for (i <- 0 until nSamples; j <- 0 until nAggregations) {
        baseArray.update(i, j, aggregators(j).copy())
      }

      val result = vds.rdd.treeAggregate(baseArray)({ case (arr, (v, (va, gs))) =>
        aggregatorA(0) = v
        aggregatorA(1) = va
        gs.iterator
          .zipWithIndex
          .foreach { case (g, i) =>
            aggregatorA(2) = localSamplesBc.value(i)
            aggregatorA(3) = localAnnotationsBc.value(i)

            for (j <- 0 until nAggregations) {
              arr(i, j).seqOp(g)
            }
          }

        arr
      }, { case (arr1, arr2) =>
        for (i <- 0 until nSamples; j <- 0 until nAggregations) {
          arr1(i, j).combOp(arr2(i, j))
        }
        arr1
      }, depth = depth)

      val sampleIndex = vds.sampleIds.zipWithIndex.toMap
      Some((s: String) => {
        val i = sampleIndex(s)
        for (j <- 0 until nAggregations) {
          aggregatorA(aggregators(j).idx) = result(i, j).result
        }
      })
    }
  }

  def makeFunctions(ec: EvalContext): (Array[Aggregator], (Array[Aggregator], (Any, Any)) => Array[Aggregator],
    (Array[Aggregator], Array[Aggregator]) => Array[Aggregator], (Array[Aggregator]) => Unit) = {

    val aggregators = ec.aggregationFunctions.toArray

    val arr = ec.a

    val baseArray = Array.fill[Aggregator](aggregators.length)(null)

    val zero = {
      for (i <- baseArray.indices)
        baseArray(i) = aggregators(i).copy()
      baseArray
    }

    val seqOp = (array: Array[Aggregator], b: (Any, Any)) => {
      val (aggT, annotation) = b
      ec.set(0, annotation)
      for (i <- array.indices) {
        array(i).seqOp(aggT)
      }
      array
    }

    val combOp = (arr1: Array[Aggregator], arr2: Array[Aggregator]) => {
      for (i <- arr1.indices) {
        arr1(i).combOp(arr2(i))
      }
      arr1
    }

    val resultOp = (array: Array[Aggregator]) => array.foreach { res => arr(res.idx) = res.result }

    (zero, seqOp, combOp, resultOp)
  }
}

class CountAggregator(aggF: (Any) => Option[Any], localIdx: Int) extends TypedAggregator[Long] {
  def zero: Long = 0L

  var _state = zero

  override def result = _state

  override def seqOp(x: Any) = aggF(x).foreach { _ => _state += 1 }

  override def combOp(agg2: TypedAggregator[Any]): Unit = _state += agg2.asInstanceOf[CountAggregator]._state

  override def copy() = new CountAggregator(aggF, localIdx)

  override def idx = localIdx
}

class FractionAggregator(aggF: (Any) => Option[Any], localIdx: Int, localA: ArrayBuffer[Any], bodyFn: () => Any, lambdaIdx: Int) extends TypedAggregator[Option[Double]] {
  var _num = 0L
  var _denom = 0L

  override def result = {
    divOption(_num.toDouble, _denom)
  }

  override def seqOp(x: Any) = {
    aggF(x).foreach { value =>
      localA(lambdaIdx) = value
      val numToAdd = if (bodyFn().asInstanceOf[Boolean]) 1 else 0
      _num += numToAdd
      _denom += 1
    }
  }

  override def combOp(agg2: TypedAggregator[Any]) = {
    val fracAgg = agg2.asInstanceOf[FractionAggregator]
    _num += fracAgg._num
    _denom += fracAgg._denom
  }

  override def copy() = new FractionAggregator(aggF, localIdx, localA, bodyFn, lambdaIdx)

  override def idx = localIdx
}

class StatAggregator(aggF: (Any) => Option[Any], localIdx: Int) extends TypedAggregator[StatCounter] {
  def zero: StatCounter = new StatCounter()

  var _state = zero

  override def result = _state

  override def seqOp(x: Any) = aggF(x).foreach(x => _state = _state.merge(DoubleNumericConversion.to(x)))

  override def combOp(agg2: TypedAggregator[Any]) = _state = _state.merge(agg2.asInstanceOf[StatAggregator]._state)

  override def copy() = new StatAggregator(aggF, localIdx)

  override def idx = localIdx
}

class HistAggregator(aggF: (Any) => Option[Any], localIdx: Int, indices: Array[Double]) extends TypedAggregator[HistogramCombiner] {
  def zero = new HistogramCombiner(indices)

  var _state = zero

  override def result = _state

  override def seqOp(x: Any) = aggF(x).foreach(x => _state = _state.merge(DoubleNumericConversion.to(x)))

  override def combOp(agg2: TypedAggregator[Any]) = _state = _state.merge(agg2.asInstanceOf[HistAggregator]._state)

  override def copy() = new HistAggregator(aggF, localIdx, indices)

  override def idx = localIdx
}

class CollectAggregator(aggF: (Any) => Option[Any], localIdx: Int) extends TypedAggregator[ArrayBuffer[Any]] {
  def zero = new ArrayBuffer[Any]

  var _state = zero

  override def result = _state

  override def seqOp(x: Any) = aggF(x).foreach(elem => _state += elem)

  override def combOp(agg2: TypedAggregator[Any]) = _state ++= agg2.asInstanceOf[CollectAggregator]._state

  override def copy() = new CollectAggregator(aggF, localIdx)

  override def idx = localIdx
}

class InfoScoreAggregator(aggF: (Any) => Option[Any], localIdx: Int) extends TypedAggregator[InfoScoreCombiner] {
  def zero = new InfoScoreCombiner()

  var _state = zero

  override def result = _state

  override def seqOp(x: Any) = aggF(x).foreach(x => _state = _state.merge(x.asInstanceOf[Genotype]))

  override def combOp(agg2: TypedAggregator[Any]) = _state = _state.merge(agg2.asInstanceOf[InfoScoreAggregator]._state)

  override def copy() = new InfoScoreAggregator(aggF, localIdx)

  override def idx = localIdx
}

class SumAggregator(aggF: (Any) => Option[Any], localIdx: Int) extends TypedAggregator[Double] {
  def zero = 0d

  var _state = zero

  override def result = _state

  override def seqOp(x: Any) = aggF(x).foreach(elem => _state += DoubleNumericConversion.to(elem))

  override def combOp(agg2: TypedAggregator[Any]) = _state += agg2.asInstanceOf[SumAggregator]._state

  override def copy() = new SumAggregator(aggF, localIdx)

  override def idx = localIdx
}

class SumArrayAggregator(aggF: (Any) => Option[Any], localIdx: Int, localPos: Position) extends TypedAggregator[IndexedSeq[Double]] {
  def zero = Array.empty[Double]

  var _state = zero

  override def result = _state.toIndexedSeq

  override def seqOp(x: Any) = aggF(x)
    .map(_.asInstanceOf[IndexedSeq[_]])
    .foreach { arr =>
      val cast = arr.map(a => if (a == null) 0d else DoubleNumericConversion.to(a))
      if (_state.isEmpty)
        _state = cast.toArray
      else {
        if (_state.length != cast.length)
          ParserUtils.error(localPos,
            s"""cannot aggregate arrays of unequal length with `sum'
                |  Found conflicting arrays of size (${ _state.size }) and (${ cast.size })""".stripMargin)
        for (i <- _state.indices)
          _state(i) += cast(i)
      }
    }

  override def combOp(agg2: TypedAggregator[Any]) = {
    val agg2result = agg2.asInstanceOf[SumArrayAggregator]._state
    if (_state.length != agg2result.length)
      ParserUtils.error(localPos,
        s"""cannot aggregate arrays of unequal length with `sum'
            |  Found conflicting arrays of size (${ _state.size }) and (${ agg2result.size })""".stripMargin)
    for (i <- _state.indices)
      _state(i) += agg2result(i)
  }

  override def copy() = new SumArrayAggregator(aggF, localIdx, localPos)

  override def idx = localIdx
}