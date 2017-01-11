package org.broadinstitute.hail.methods

import java.io.{ObjectInputStream, ObjectOutputStream}

import org.apache.spark.util.StatCounter
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.driver.HailConfiguration
import org.broadinstitute.hail.expr.{TAggregable, _}
import org.broadinstitute.hail.stats._
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object Aggregators {

  def variantEC(vds: VariantDataset): EvalContext = {
    val aggregationST = Map(
      "global" -> (0, vds.globalSignature),
      "v" -> (1, TVariant),
      "va" -> (2, vds.vaSignature),
      "g" -> (3, TGenotype),
      "s" -> (4, TSample),
      "sa" -> (5, vds.saSignature))
    EvalContext(Map(
      "global" -> (0, vds.globalSignature),
      "v" -> (1, TVariant),
      "va" -> (2, vds.vaSignature),
      "gs" -> (3, TAggregable(TGenotype, aggregationST))))
  }

  def buildVariantAggregations(vds: VariantDataset, ec: EvalContext): Option[(Variant, Annotation, Iterable[Genotype]) => Unit] = {

    val aggregations = ec.aggregations
    if (aggregations.isEmpty)
      return None

    val localA = ec.a
    val localSamplesBc = vds.sampleIdsBc
    val localAnnotationsBc = vds.sampleAnnotationsBc
    val localGlobalAnnotations = vds.globalAnnotation

    Some({ (v: Variant, va: Annotation, gs: Iterable[Genotype]) =>
      val aggs = aggregations.map { case (_, _, agg0) => agg0.copy() }
      localA(0) = localGlobalAnnotations
      localA(1) = v
      localA(2) = va

      val gsIt = gs.iterator
      var i = 0
      // gsIt assume hasNext is always called before next
      while (gsIt.hasNext) {
        localA(3) = gsIt.next
        localA(4) = localSamplesBc.value(i)
        localA(5) = localAnnotationsBc.value(i)

        var j = 0
        while (j < aggs.size) {
          aggregations(j)._2(aggs(j).seqOp)
          j += 1
        }

        i += 1
      }

      i = 0
      while (i < aggs.size) {
        localA(aggregations(i)._1) = aggs(i).result
        i += 1
      }
    })
  }

  def sampleEC(vds: VariantDataset): EvalContext = {
    val aggregationST = Map(
      "global" -> (0, vds.globalSignature),
      "s" -> (1, TSample),
      "va" -> (2, vds.saSignature),
      "g" -> (3, TGenotype),
      "v" -> (4, TVariant),
      "va" -> (5, vds.vaSignature))
    EvalContext(Map(
      "global" -> (0, vds.globalSignature),
      "s" -> (1, TSample),
      "sa" -> (2, vds.saSignature),
      "gs" -> (3, TAggregable(TGenotype, aggregationST))))
  }

  def buildSampleAggregations(vds: VariantDataset, ec: EvalContext): Option[(String) => Unit] = {

    val aggregations = ec.aggregations

    if (aggregations.isEmpty)
      return None

    val localA = ec.a
    val localGlobalAnnotation = vds.globalAnnotation
    val localSamplesBc = vds.sampleIdsBc
    val localSampleAnnotationsBc = vds.sampleAnnotationsBc

    val nAggregations = aggregations.length
    val nSamples = vds.nSamples
    val depth = HailConfiguration.treeAggDepth(vds.nPartitions)

    val baseArray = MultiArray2.fill[Aggregator](nSamples, nAggregations)(null)
    for (i <- 0 until nSamples; j <- 0 until nAggregations) {
      baseArray.update(i, j, aggregations(j)._3.copy())
    }

    val result = vds.rdd.treeAggregate(baseArray)({ case (arr, (v, (va, gs))) =>
      localA(0) = localGlobalAnnotation
      localA(4) = v
      localA(5) = va

      val gsIt = gs.iterator
      var i = 0
      // gsIt assume hasNext is always called before next
      while (gsIt.hasNext) {
        localA(1) = localSamplesBc.value(i)
        localA(2) = localSampleAnnotationsBc.value(i)
        localA(3) = gsIt.next

        var j = 0
        while (j < nAggregations) {
          aggregations(j)._2(arr(i, j).seqOp)
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
        localA(aggregations(j)._1) = result(i, j).result
      }
    })
  }

  def makeFunctions[T](ec: EvalContext, setEC: (EvalContext, T) => Unit): (Array[Aggregator],
    (Array[Aggregator], T) => Array[Aggregator],
    (Array[Aggregator], Array[Aggregator]) => Array[Aggregator],
    (Array[Aggregator] => Unit)) = {

    val aggregations = ec.aggregations

    val localA = ec.a

    val zVal = aggregations.map { case (_, _, agg0) => agg0.copy() }.toArray

    val seqOp = (array: Array[Aggregator], t: T) => {
      setEC(ec, t)
      for (i <- array.indices) {
        aggregations(i)._2(array(i).seqOp)
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

    val resultOp = (aggs: Array[Aggregator]) =>
      (aggs, aggregations).zipped.foreach { case (agg, (idx, _, _)) => localA(idx) = agg.result }

    (zVal, seqOp, combOp, resultOp)
  }
}

class CountAggregator() extends TypedAggregator[Long] {

  var _state = 0L

  def result = _state

  def seqOp(x: Any) {
    _state += 1
  }

  def combOp(agg2: this.type) {
    _state += agg2._state
  }

  def copy() = new CountAggregator()
}

class FractionAggregator(f: (Any) => Any)
  extends TypedAggregator[java.lang.Double] {

  var _num = 0L
  var _denom = 0L

  def result =
    if (_denom == 0L)
      null
    else
      _num.toDouble / _denom

  def seqOp(x: Any) {
    val r = f(x)
    _denom += 1
    if (r.asInstanceOf[Boolean])
      _num += 1
    }
  }

  def combOp(agg2: this.type) {
    _num += agg2._num
    _denom += agg2._denom
  }

  def copy() = new FractionAggregator(f)
}

class StatAggregator() extends TypedAggregator[Annotation] {

  var _state = new StatCounter()

  def result =
    if (_state.count == 0)
      null
    else
      Annotation(_state.mean, _state.stdev, _state.min, _state.max, _state.count, _state.sum)

  def seqOp(x: Any) {
    if (x != null)
      _state.merge(DoubleNumericConversion.to(x))
  }

  def combOp(agg2: this.type) {
    _state.merge(agg2._state)
  }

  def copy() = new StatAggregator()
}

class CounterAggregator extends TypedAggregator[IndexedSeq[Annotation]] {
  var m = new mutable.HashMap[Any, Long]

  def result = m.map { case (k, v) =>
    Annotation(k, v)
  }.toArray[Annotation]: IndexedSeq[Annotation]

  def seqOp(x: Any) {
    m.updateValue(x, 0L, _ + 1)
  }

  def combOp(agg2: this.type) {
    agg2.m.foreach { case (k, v) =>
      m.updateValue(k, 0L, _ + v)
    }
  }

  def copy() = new CounterAggregator()
}

class HistAggregator(indices: Array[Double])
  extends TypedAggregator[Annotation] {

  var _state = new HistogramCombiner(indices)

  def result = _state.toAnnotation

  def seqOp(x: Any) {
    if (x != null)
      _state.merge(DoubleNumericConversion.to(x))
  }

  def combOp(agg2: this.type) {
    _state.merge(agg2._state)
  }

  def copy() = new HistAggregator(indices)
}

class CollectAggregator extends TypedAggregator[ArrayBuffer[Any]] {

  var _state = new ArrayBuffer[Any]

  def result = _state

  def seqOp(x: Any) {
    _state += x
  }

  def combOp(agg2: this.type) = _state ++= agg2._state

  def copy() = new CollectAggregator()
}

class InfoScoreAggregator extends TypedAggregator[Annotation] {

  var _state = new InfoScoreCombiner()

  def result = _state.asAnnotation

  def seqOp(x: Any) {
    if (x != null)
      _state.merge(x.asInstanceOf[Genotype])
  }

  def combOp(agg2: this.type) {
    _state.merge(agg2._state)
  }

  def copy() = new InfoScoreAggregator()
}

class HWEAggregator() extends TypedAggregator[Annotation] {

  var _state = new HWECombiner()

  def result = _state.asAnnotation

  def seqOp(x: Any) {
    if (x != null)
      _state.merge(x.asInstanceOf[Genotype])
  }

  def combOp(agg2: this.type) {
    _state.merge(agg2._state)
  }

  def copy() = new HWEAggregator()
}

class SumAggregator[T](implicit ev: scala.math.Numeric[T]) extends TypedAggregator[T] {

  import scala.math.Numeric.Implicits._

  var _state: T = ev.zero

  def result = _state

  def seqOp(x: Any) {
    if (x != null)
      _state += x.asInstanceOf[T]
  }

  def combOp(agg2: this.type) = _state += agg2._state

  def copy() = new SumAggregator()
}

class SumArrayAggregator[T](implicit ev: scala.math.Numeric[T], ct: ClassTag[T])
  extends TypedAggregator[IndexedSeq[T]] {

  import scala.math.Numeric.Implicits._

  var _state: Array[T] = _

  def result = _state

  def seqOp(x: Any) {
    if (x != null) {
      val r = x.asInstanceOf[IndexedSeq[T]]
      if (_state == null)
        _state = r.map(x => if (x == null) ev.zero else x).toArray
      else {
        if (r.length != _state.length)
          fatal(
            s"""cannot aggregate arrays of unequal length with `sum'
                |Found conflicting arrays of size (${ _state.length }) and (${ r.length })""".stripMargin)
        else {
          var i = 0
          while (i < _state.length) {
            if (r(i) != null)
              _state(i) += r(i)
            i += 1
          }
        }
      }
    }
  }

  def combOp(agg2: this.type) = {
    val agg2state = agg2._state
    if (_state == null)
      _state = agg2._state
    else if (agg2._state != null) {
      if (_state.length != agg2state.length)
        fatal(
          s"""cannot aggregate arrays of unequal length with `sum'
              |  Found conflicting arrays of size (${ _state.length }) and (${ agg2state.length })""".
            stripMargin)
      for (i <- _state.indices)
        _state(i) += agg2state(i)
    }
  }

  def copy() = new SumArrayAggregator()
}

class CallStatsAggregator(variantF: (Any) => Any)
  extends TypedAggregator[Annotation] {

  var first = true
  var combiner: CallStatsCombiner = _

  def result =
    if (combiner != null)
      combiner.result().asAnnotation
    else
      null

  def seqOp(x: Any) {
    if (first) {
      first = false

      val v = variantF(x)
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

  def copy() = new CallStatsAggregator(variantF)
}

class InbreedingAggregator(getAF: (Genotype) => Any) extends TypedAggregator[Annotation] {

  var _state = new InbreedingCombiner()

  def result = _state.asAnnotation

  def seqOp(x: Any) = {
    if (x != null) {
      val g = x.asInstanceOf[Genotype]
      val af = getAF(g)
      if (af != null)
        _state.merge(x.asInstanceOf[Genotype], af.asInstanceOf[Double])
    }
  }

  def combOp(agg2: this.type) = _state.merge(agg2.asInstanceOf[InbreedingAggregator]._state)

  def copy() = new InbreedingAggregator(getAF)
}

class TakeAggregator(n: Int) extends TypedAggregator[IndexedSeq[Any]] {
  var _state = new ArrayBuffer[Any]()

  def result = _state.toArray[Any]: IndexedSeq[Any]

  def seqOp(x: Any) = {
    if (_state.length < n)
      _state += x
  }

  def combOp(agg2: this.type) {
    agg2._state.foreach(seqOp)
  }

  def copy() = new TakeAggregator(n)
}

class TakeByAggregator[T](var f: (Any) => Any, var n: Int)(implicit var tord: Ordering[T]) extends TypedAggregator[IndexedSeq[Any]] {
  def this() = this(null, 0)(null)

  def makeOrd(): Ordering[(Any, Any)] = if (tord != null) {
    new Ordering[(Any, Any)] {
      // nulls are the largest
      def compare(a: (Any, Any), b: (Any, Any)) = (a._2, b._2) match {
        case (null, null) => 0
        case (null, _) => 1
        case (_, null) => -1
        case (x, y) => -tord.compare(x.asInstanceOf[T], y.asInstanceOf[T])
      }
    }
  } else
    null

  // double-reverse makes nulls the smallest
  var ord: Ordering[(Any, Any)] = makeOrd()

  // PriorityQueue is not serializable
  // https://issues.scala-lang.org/browse/SI-7568
  // fixed in Scala 2.11.0-M7
  var _state = if (ord != null)
      new mutable.PriorityQueue[(Any, Any)]()(ord)
    else
      null

  def result = _state.toArray[(Any, Any)].map(_._1).reverse: IndexedSeq[Any]

  def seqOp(x: Any) = {
    val p = (x, f(x))
    if (_state.length < n)
      _state += p
    else {
      if (ord.compare(p, _state.head) < 0) {
        _state.dequeue()
        _state += p
      }
    }
  }

  def combOp(agg2: this.type) {
    agg2._state.foreach(seqOp)
  }

  def copy() = new TakeByAggregator(f, n)

  private def writeObject(oos: ObjectOutputStream) {
    oos.writeObject(f)
    oos.writeInt(n)
    oos.writeObject(tord)
    oos.writeObject(_state.toArray[(Any, Any)])
  }

  private def readObject(ois: ObjectInputStream) {
    f = ois.readObject().asInstanceOf[(Any) => Any]
    n = ois.readInt()
    tord = ois.readObject().asInstanceOf[Ordering[T]]
    ord = makeOrd()

    val elems = ois.readObject().asInstanceOf[Array[(Any, Any)]]
    _state = mutable.PriorityQueue[(Any, Any)](elems: _*)(ord)
  }
}
