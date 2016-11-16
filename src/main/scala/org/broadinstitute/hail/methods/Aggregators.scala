package org.broadinstitute.hail.methods

import org.apache.spark.util.StatCounter
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.driver.HailConfiguration
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.stats._
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.parsing.input.Position

object Aggregators {

  def buildVariantAggregations(vds: VariantDataset, ec: EvalContext): Option[(Variant, Annotation, Iterable[Genotype]) => Unit] = {

    val aggFunctions = ec.aggregationFunctions.map(_._1)
    val aggregators = ec.aggregationFunctions.map(_._2)

    val aggregatorA = ec.a

    if (aggregators.nonEmpty) {

      val localSamplesBc = vds.sampleIdsBc
      val localAnnotationsBc = vds.sampleAnnotationsBc

      val f = (v: Variant, va: Annotation, gs: Iterable[Genotype]) => {
        val baseArray = aggregators.map(_.copy())
        aggregatorA(0) = v
        aggregatorA(1) = va
        (gs, localSamplesBc.value, localAnnotationsBc.value).zipped
          .foreach { case (g, s, sa) =>
            aggregatorA(2) = s
            aggregatorA(3) = sa
            (baseArray, aggFunctions).zipped.foreach { case (agg, aggF) =>
              agg.seqOp(aggF(g))
            }
          }

        baseArray.foreach { agg => aggregatorA(agg.idx) = agg.result }
      }
      Some(f)
    } else
      None
  }

  def buildSampleAggregations(vds: VariantDataset, ec: EvalContext): Option[(String) => Unit] = {
    val aggFunctions = ec.aggregationFunctions.map(_._1)
    val aggregators = ec.aggregationFunctions.map(_._2)
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
        var i = 0
        gs.foreach { g =>
          aggregatorA(2) = localSamplesBc.value(i)
          aggregatorA(3) = localAnnotationsBc.value(i)

          var j = 0
          while (j < nAggregations) {
            arr(i, j).seqOp(aggFunctions(j)(g))
            j += 1
          }
          i += 1
        }
        arr
      }, { case (arr1, arr2) =>
        for (i <- 0 until nSamples; j <- 0 until nAggregations) {
          val a1 = arr1(i, j)
          a1.combOp(arr2(i, j).asInstanceOf[a1.type])
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

    val aggFunctions = ec.aggregationFunctions.map(_._1)
    val aggregators = ec.aggregationFunctions.map(_._2)

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
        array(i).seqOp(aggFunctions(i)(aggT))
      }
      array
    }

    val combOp = (arr1: Array[Aggregator], arr2: Array[Aggregator]) => {
      for (i <- arr1.indices) {
        val a1 = arr1(i)
        a1.combOp(arr2(i).asInstanceOf[a1.type])
      }
      arr1
    }

    val resultOp = (array: Array[Aggregator]) => array.foreach { res => arr(res.idx) = res.result }

    (zero, seqOp, combOp, resultOp)
  }
}

class CountAggregator(val idx: Int) extends TypedAggregator[Long] {

  var _state = 0L

  def result = _state

  override def seqOp(x: Any) {
    if (x != null)
      _state += 1
  }

  def combOp(agg2: this.type) {
    _state += agg2._state
  }

  override def copy() = new CountAggregator(idx)
}

class FractionAggregator(val idx: Int, localA: ArrayBuffer[Any], bodyFn: () => Any, lambdaIdx: Int)
  extends TypedAggregator[java.lang.Double] {

  var _num = 0L
  var _denom = 0L

  def result =
    if (_denom == 0L)
      null
    else
      _num.toDouble / _denom

  override def seqOp(x: Any) {
    if (x != null) {
      _denom += 1
      localA(lambdaIdx) = x
      if (bodyFn().asInstanceOf[Boolean])
        _num += 1
    }
  }

  def combOp(agg2: this.type) {
    _num += agg2._num
    _denom += agg2._denom
  }

  override def copy() = new FractionAggregator(idx, localA, bodyFn, lambdaIdx)
}

class StatAggregator(val idx: Int) extends TypedAggregator[StatCounter] {

  var _state = new StatCounter()

  def result = _state

  override def seqOp(x: Any) {
    if (x != null)
      _state.merge(DoubleNumericConversion.to(x))
  }

  def combOp(agg2: this.type) {
    _state.merge(agg2._state)
  }

  override def copy() = new StatAggregator(idx)
}

class CounterAggregator(val idx: Int) extends TypedAggregator[mutable.HashMap[Any, Long]] {
  var m = new mutable.HashMap[Any, Long]

  def result = m

  def seqOp(x: Any) {
    val r = f(x)
    if (r != null)
      m.updateValue(r, 0L, _ + 1)
  }

  def combOp(agg2: this.type) {
    agg2.m.foreach { case (k, v) =>
      m.updateValue(k, 0L, _ + v)
    }
  }

  override def copy() = new StatAggregator(idx)
}

class HistAggregator(val idx: Int, indices: Array[Double])
  extends TypedAggregator[HistogramCombiner] {

  var _state = new HistogramCombiner(indices)

  def result = _state

  override def seqOp(x: Any) {
    if (x != null)
      _state.merge(DoubleNumericConversion.to(x))
  }

  def combOp(agg2: this.type) {
    _state.merge(agg2._state)
  }

  override def copy() = new HistAggregator(idx, indices)
}

class CollectAggregator(val idx: Int) extends TypedAggregator[ArrayBuffer[Any]] {

  var _state = new ArrayBuffer[Any]

  def result = _state

  override def seqOp(x: Any) {
    if (x != null)
      _state += x
  }

  def combOp(agg2: this.type) = _state ++= agg2._state

  override def copy() = new CollectAggregator(idx)
}

class InfoScoreAggregator(val idx: Int) extends TypedAggregator[InfoScoreCombiner] {

  var _state = new InfoScoreCombiner()

  def result = _state

  override def seqOp(x: Any) {
    if (x != null)
      _state.merge(x.asInstanceOf[Genotype])
  }

  def combOp(agg2: this.type) {
    _state.merge(agg2._state)
  }

  override def copy() = new InfoScoreAggregator(idx)
}

class HWEAggregator(val idx: Int) extends TypedAggregator[HWECombiner] {

  var _state = new HWECombiner()

  def result = _state

  override def seqOp(x: Any) {
    if (x != null)
      _state.merge(x.asInstanceOf[Genotype])
  }

  def combOp(agg2: this.type) {
    _state.merge(agg2._state)
  }

  override def copy() = new HWEAggregator(idx)
}

class SumAggregator(val idx: Int) extends TypedAggregator[Double] {
  var _state = 0d

  def result = _state

  override def seqOp(x: Any) {
    if (x != null)
      _state += DoubleNumericConversion.to(x)
  }

  def combOp(agg2: this.type) = _state += agg2._state

  override def copy() = new SumAggregator(idx)
}

class SumArrayAggregator(val idx: Int, localPos: Position)
  extends TypedAggregator[IndexedSeq[Double]] {

  var _state: Array[Double] = _

  def result = _state

  override def seqOp(x: Any) {
    val r = x.asInstanceOf[IndexedSeq[Any]]
    if (r != null) {
      if (_state == null)
        _state = r.map(x => if (x == null) 0d else DoubleNumericConversion.to(x)).toArray
      else {
        if (r.length != _state.length)
          ParserUtils.error(localPos,
            s"""cannot aggregate arrays of unequal length with `sum'
                |Found conflicting arrays of size (${ _state.length }) and (${ r.length })""".stripMargin)
        else {
          var i = 0
          while (i < _state.length) {
            if (r(i) != null)
              _state(i) += DoubleNumericConversion.to(r(i))
            i += 1
          }
        }
      }
    }
  }

  def combOp(agg2: this.type) = {
    val agg2state = agg2._state
    if (_state.length != agg2state.length)
      ParserUtils.error(localPos,
        s"""cannot aggregate arrays of unequal length with `sum'
            |  Found conflicting arrays of size (${ _state.length }) and (${ agg2state.length })""".stripMargin)
    for (i <- _state.indices)
      _state(i) += agg2state(i)
  }

  override def copy() = new SumArrayAggregator(idx, localPos)
}

class CallStatsAggregator(val idx: Int, variantF: () => Any)
  extends TypedAggregator[CallStats] {

  var first = true
  var combiner: CallStatsCombiner = _

  def result: CallStats =
    if (combiner != null)
      combiner.result()
    else
      null

  def seqOp(x: Any) {
    if (first) {
      first = false

      val v = variantF()
      if (v != null)
        combiner = new CallStatsCombiner(v.asInstanceOf[Variant])
    }

    if (combiner != null) {
      if (x != null)
        combiner.merge(x.asInstanceOf[Genotype])
    }
  }

  def merge(x: Genotype) {
    combiner.merge(x)
  }

  def combOp(agg2: this.type) {
    combiner.merge(agg2.combiner)
  }

  def copy(): TypedAggregator[CallStats] = new CallStatsAggregator(idx, variantF)
}

class InbreedingAggregator(localIdx: Int, getAF: () => Any) extends TypedAggregator[InbreedingCombiner] {

  var _state = new InbreedingCombiner()

  def result = _state

  override def seqOp(x: Any) = {
    val af = getAF()

    if (x != null && af != null)
      _state.merge(x.asInstanceOf[Genotype], DoubleNumericConversion.to(af))
  }

  def combOp(agg2: this.type) = _state.merge(agg2.asInstanceOf[InbreedingAggregator]._state)

  override def copy() = new InbreedingAggregator(localIdx, getAF)

  def idx = localIdx
}
