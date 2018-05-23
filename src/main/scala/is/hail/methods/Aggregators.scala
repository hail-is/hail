package is.hail.methods

import java.io.{ObjectInputStream, ObjectOutputStream}

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr._
import is.hail.expr.types._
import is.hail.stats._
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.SparkContext
import org.apache.spark.util.StatCounter

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object Aggregators {

  def buildRowAggregationsByKey(vsm: MatrixTable, nKeys: Int, keyMap: Array[Int], ec: EvalContext): (RegionValue) => Array[() => Unit] =
    buildRowAggregationsByKey(vsm.sparkContext, vsm.matrixType, vsm.value.globals, vsm.value.colValues, nKeys, keyMap, ec)

  def buildRowAggregationsByKey(sc: SparkContext,
    typ: MatrixType,
    globals: BroadcastRow,
    colValues: BroadcastIndexedSeq,
    nKeys: Int,
    keyMap: Array[Int],
    ec: EvalContext): (RegionValue) => Array[() => Unit] = {

    val aggregations = ec.aggregations
    if (aggregations.isEmpty)
      return { rv => Array.fill[() => Unit](nKeys) { () => Unit } }

    val localA = ec.a
    val localNCols = colValues.value.length
    val localAnnotationsBc = colValues.broadcast
    val globalsBc = globals.broadcast

    val fullRowType = typ.rvRowType
    val localEntriesIndex = typ.entriesIdx

    { (rv: RegionValue) =>
      val fullRow = new UnsafeRow(fullRowType, rv)

      val aggs = MultiArray2.fill[Aggregator](nKeys, aggregations.size)(null)
      var nk = 0
      while (nk < nKeys) {
        var nagg = 0
        while (nagg < aggregations.size) {
          aggs.update(nk, nagg, aggregations(nagg)._3.copy())
          nagg += 1
        }
        nk += 1
      }
      localA(0) = globalsBc.value
      localA(1) = fullRow
      val is = fullRow.getAs[IndexedSeq[Annotation]](localEntriesIndex)

      var i = 0
      while (i < localNCols) {
        localA(2) = is(i)
        if (keyMap(i) != -1) {
          localA(3) = localAnnotationsBc.value(i)

          var j = 0
          while (j < aggs.n2) {
            aggregations(j)._2(aggs(keyMap(i), j).seqOp)
            j += 1
          }
        }
        i += 1
      }
      Array.tabulate[() => Unit](nKeys) { k => { () => {
        var j = 0
        while (j < aggs.n2) {
          aggregations(j)._1.v = aggs(k, j).result
          j += 1
        }
      }}}
    }
  }

  def buildRowAggregations(vsm: MatrixTable, ec: EvalContext): Option[(RegionValue) => Unit] =
    buildRowAggregations(vsm.sparkContext, vsm.matrixType, vsm.value.globals, vsm.value.colValues, ec)

  def buildRowAggregations(sc: SparkContext,
    typ: MatrixType,
    globals: BroadcastRow,
    colValues: BroadcastIndexedSeq,
    ec: EvalContext): Option[(RegionValue) => Unit] = {

    val aggregations = ec.aggregations
    if (aggregations.isEmpty)
      return None

    val localA = ec.a
    val localNCols = colValues.value.length
    val colValuesBc = colValues.broadcast
    val fullRowType = typ.rvRowType
    val localEntriesIndex = typ.entriesIdx
    val globalsBc = globals.broadcast

    Some({ (rv: RegionValue) =>

      val fullRow = new UnsafeRow(fullRowType, rv)

      val aggs = aggregations.map { case (_, _, agg0) => agg0.copy() }

      ec.set(0, globalsBc.value)
      ec.set(1, fullRow)

      val is = fullRow.getAs[IndexedSeq[Annotation]](localEntriesIndex)
      var i = 0

      while (i < localNCols) {
        ec.set(2, is(i))
        ec.set(3, colValuesBc.value(i))

        var j = 0
        while (j < aggs.size) {
          aggregations(j)._2(aggs(j).seqOp)
          j += 1
        }

        i += 1
      }

      i = 0
      while (i < aggs.size) {
        aggregations(i)._1.v = aggs(i).result
        i += 1
      }
    })
  }

  def buildColAggregations(hc: HailContext, value: MatrixValue, ec: EvalContext): Option[(Int) => Unit] = {

    val aggregations = ec.aggregations

    if (aggregations.isEmpty)
      return None

    val localA = ec.a
    val localNCols = value.nCols
    val globalsBc = value.globals.broadcast
    val localColValuesBc = value.colValues.broadcast

    val nAggregations = aggregations.length
    val nCols = value.nCols
    val depth = treeAggDepth(hc, value.nPartitions)

    val baseArray = MultiArray2.fill[Aggregator](nCols, nAggregations)(null)
    for (i <- 0 until nCols; j <- 0 until nAggregations) {
      baseArray.update(i, j, aggregations(j)._3.copy())
    }

    val fullRowType = value.typ.rvRowType
    val localEntriesIndex = value.typ.entriesIdx

    val result = value.rvd.treeAggregate(baseArray)({ case (arr, rv) =>
      val fullRow = new UnsafeRow(fullRowType, rv)

      localA(0) = globalsBc.value
      localA(3) = fullRow

      val gs = fullRow.getAs[IndexedSeq[Annotation]](localEntriesIndex)

      var i = 0
      while (i < localNCols) {
        localA(1) = localColValuesBc.value(i)
        localA(2) = gs(i)

        var j = 0
        while (j < nAggregations) {
          aggregations(j)._2(arr(i, j).seqOp)
          j += 1
        }
        i += 1
      }

      // clean up
      localA.indices.foreach { i =>
        localA(i) = null
      }

      arr
    }, { case (arr1, arr2) =>
      for (i <- 0 until nCols; j <- 0 until nAggregations) {
        val a1 = arr1(i, j)
        a1.combOp(arr2(i, j).asInstanceOf[a1.type])
      }
      arr1
    }, depth = depth)

    Some((i: Int) => {
      for (j <- 0 until nAggregations) {
        aggregations(j)._1.v = result(i, j).result
      }
    })
  }

  def makeColFunctions(vsm: MatrixTable, aggExpr: String): ColFunctions = {
    val ec = vsm.colEC

    val (resultNames, resultTypes, aggF) = Parser.parseNamedExprs(aggExpr, ec)

    val newType = TStruct(resultNames.zip(resultTypes): _*)

    val localNCols = vsm.numCols
    val localColValuesBc = vsm.colValues.broadcast

    val aggregations = ec.aggregations
    val nAggregations = aggregations.size

    val globalsBc = vsm.globals.broadcast

    val ma = MultiArray2.fill[Aggregator](localNCols, nAggregations)(null)
    val zVal = { () =>
      for (i <- 0 until localNCols; j <- 0 until nAggregations) {
        ma.update(i, j, aggregations(j)._3.copy())
      }
      ma
    }

    val fullRowType = vsm.rvRowType
    val localEntriesIndex = vsm.entriesIndex


    val seqOp = (ma: MultiArray2[Aggregator], rv: RegionValue) => {
      val fullRow = new UnsafeRow(fullRowType, rv)

      val is = fullRow.getAs[IndexedSeq[Annotation]](localEntriesIndex)

      var i = 0
      while (i < localNCols) {
        ec.setAll(globalsBc.value,
          localColValuesBc.value(i),
          is(i),
          fullRow)

        var j = 0
        while (j < nAggregations) {
          aggregations(j)._2(ma(i, j).seqOp)
          j += 1
        }
        i += 1
      }
      ma
    }

    val resultOp = (ma: MultiArray2[Aggregator], rvb: RegionValueBuilder) => {
      ec.set(0, globalsBc.value)
      rvb.startArray(localNCols)

      var i = 0
      while (i < localNCols) {
        ec.set(2, localColValuesBc.value(i))
        var j = 0
        while (j < nAggregations) {
          aggregations(j)._1.v = ma(i, j).result
          j += 1
        }
        rvb.startStruct()
        val fields = aggF()
        var k = 0
        while (k < fields.size) {
          rvb.addAnnotation(newType.types(k), fields(k))
          k += 1
        }
        rvb.endStruct()
        i += 1
      }
      rvb.endArray()
    }

    ColFunctions(zVal, seqOp, resultOp, newType)
  }

  case class ColFunctions(
    zero: () => MultiArray2[Aggregator],
    seqOp: (MultiArray2[Aggregator], RegionValue) => MultiArray2[Aggregator],
    resultOp: (MultiArray2[Aggregator], RegionValueBuilder) => Unit,
    resultType: TStruct)

  def makeFunctions[T](ec: EvalContext, setEC: (EvalContext, T) => Unit): (Array[Aggregator],
    (Array[Aggregator], T) => Array[Aggregator],
    (Array[Aggregator], Array[Aggregator]) => Array[Aggregator],
    (Array[Aggregator] => Unit)) = {

    val aggregations = ec.aggregations

    val zVal = aggregations.map { case (_, _, agg0) => agg0.copy() }.toArray

    val seqOp = (array: Array[Aggregator], t: T) => {
      setEC(ec, t)
      var i = 0
      while (i < array.length) {
        aggregations(i)._2(array(i).seqOp)
        i += 1
      }
      array
    }

    val combOp = (arr1: Array[Aggregator], arr2: Array[Aggregator]) => {
      var i = 0
      while (i < arr1.length) {
        val a1 = arr1(i)
        a1.combOp(arr2(i).asInstanceOf[a1.type])
        i += 1
      }
      arr1
    }

    val resultOp = (aggs: Array[Aggregator]) =>
      (aggs, aggregations).zipped.foreach { case (agg, (b, _, _)) => b.v = agg.result }

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

  def combOp(agg2: this.type) {
    _num += agg2._num
    _denom += agg2._denom
  }

  def copy() = new FractionAggregator(f)
}

class ExistsAggregator(f: (Any) => Any)
  extends TypedAggregator[Boolean] {

  var exists: Boolean = false

  def result: Boolean = exists

  def seqOp(x: Any) {
    exists = exists || {
      val r = f(x)
      r != null && r.asInstanceOf[Boolean]
    }
  }

  def combOp(agg2: this.type) {
    exists = exists || agg2.exists
  }

  def copy() = new ExistsAggregator(f)
}

class ForallAggregator(f: (Any) => Any)
  extends TypedAggregator[Boolean] {

  var forall: Boolean = true

  def result: Boolean = forall

  def seqOp(x: Any) {
    forall = forall && {
      val r = f(x)
      r != null || r.asInstanceOf[Boolean]
    }
  }

  def combOp(agg2: this.type) {
    forall = forall && agg2.forall
  }

  def copy() = new ForallAggregator(f)
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

class CounterAggregator(t: Type) extends TypedAggregator[Map[Annotation, Long]] {
  var m = new mutable.HashMap[Any, Long]

  def result: Map[Annotation, Long] = m.toMap

  def seqOp(x: Any) {
    // FIXME only need to copy on the first one
    val cx = Annotation.copy(t, x)
    m.updateValue(cx, 0L, _ + 1)
  }

  def combOp(agg2: this.type) {
    agg2.m.foreach { case (k, v) =>
      m.updateValue(k, 0L, _ + v)
    }
  }

  def copy() = new CounterAggregator(t)
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

class CollectSetAggregator(t: Type) extends TypedAggregator[Set[Any]] {

  var _state = new mutable.HashSet[Any]

  def result = _state.toSet

  def seqOp(x: Any) {
    _state += Annotation.copy(t, x)
  }

  def combOp(agg2: this.type) = _state ++= agg2._state

  def copy() = new CollectSetAggregator(t)
}

class CollectAggregator(t: Type) extends TypedAggregator[ArrayBuffer[Any]] {

  var _state = new ArrayBuffer[Any]

  def result = _state

  def seqOp(x: Any) {
    _state += Annotation.copy(t, x)
  }

  def combOp(agg2: this.type) = _state ++= agg2._state

  def copy() = new CollectAggregator(t)
}

class InfoScoreAggregator extends TypedAggregator[Annotation] {

  var _state = new InfoScoreCombiner()

  def result = _state.asAnnotation

  def seqOp(x: Any) {
    if (x != null)
      _state.merge(x.asInstanceOf[IndexedSeq[java.lang.Double]])
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
      _state.merge(x.asInstanceOf[Call])
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

class ProductAggregator[T](implicit ev: scala.math.Numeric[T]) extends TypedAggregator[T] {

  import scala.math.Numeric.Implicits._

  var _state: T = ev.one

  def result = _state

  def seqOp(x: Any) {
    if (x != null)
      _state *= x.asInstanceOf[T]
  }

  def combOp(agg2: this.type) = _state *= agg2._state

  def copy() = new ProductAggregator()
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

class MaxAggregator[T, BoxedT >: Null](implicit ev: NumericPair[T, BoxedT], ct: ClassTag[T]) extends TypedAggregator[BoxedT] {

  import ev.numeric
  import Ordering.Implicits._

  var _state: T = ev.numeric.zero
  var _isNull: Boolean = true

  def result: BoxedT = if (_isNull) null else ev.box(_state)

  def seqOp(x: Any) {
    if (x != null && (_isNull || ev.unbox(x.asInstanceOf[BoxedT]) > _state)) {
      _isNull = false
      _state = ev.unbox(x.asInstanceOf[BoxedT])
    }
  }

  def combOp(agg2: this.type) {
    if (!agg2._isNull && (_isNull || agg2._state > _state)) {
      _isNull = false
      _state = agg2._state
    }
  }

  def copy() = new MaxAggregator[T, BoxedT]()
}

class MinAggregator[T, BoxedT >: Null](implicit ev: NumericPair[T, BoxedT], ct: ClassTag[T])
  extends TypedAggregator[BoxedT] {

  import ev.numeric
  import Ordering.Implicits._

  var _state: T = ev.numeric.zero
  var _isNull: Boolean = true

  def result: BoxedT = if (_isNull) null else ev.box(_state)

  def seqOp(x: Any) {
    if (x != null && (_isNull || ev.unbox(x.asInstanceOf[BoxedT]) < _state)) {
      _isNull = false
      _state = ev.unbox(x.asInstanceOf[BoxedT])
    }
  }

  def combOp(agg2: this.type) {
    if (!agg2._isNull && (_isNull || agg2._state < _state)) {
      _isNull = false
      _state = agg2._state
    }
  }

  def copy() = new MinAggregator[T, BoxedT]()
}

class CallStatsAggregator(nAllelesF: (Any) => Any)
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

      val nAlleles = nAllelesF(x)
      if (nAlleles != null)
        combiner = new CallStatsCombiner(nAlleles.asInstanceOf[Int])
    }

    if (combiner != null && x != null)
      combiner.merge(x.asInstanceOf[Call])
  }

  def combOp(agg2: this.type) {
    if (agg2.combiner != null) {
      if (combiner == null)
        combiner = new CallStatsCombiner(agg2.combiner.nAlleles)
      combiner.merge(agg2.combiner)
    }
  }

  def copy() = new CallStatsAggregator(nAllelesF)
}

class InbreedingAggregator(getAF: (Call) => Any) extends TypedAggregator[Annotation] {
  var _state = new InbreedingCombiner()

  def result = _state.asAnnotation

  def seqOp(x: Any) = {
    if (x != null) {
      val gt = x.asInstanceOf[Call]
      val af = getAF(gt)
      if (af != null)
        _state.merge(gt, af.asInstanceOf[Double])
    }
  }

  def combOp(agg2: this.type) = _state.merge(agg2.asInstanceOf[InbreedingAggregator]._state)

  def copy() = new InbreedingAggregator(getAF)
}

class TakeAggregator(t: Type, n: Int) extends TypedAggregator[IndexedSeq[Any]] {
  var _state = new ArrayBuffer[Any]()

  def result = _state.toArray[Any]: IndexedSeq[Any]

  def seqOp(x: Any) = {
    if (_state.length < n)
      _state += Annotation.copy(t, x)
  }

  def combOp(agg2: this.type) {
    agg2._state.foreach(seqOp)
  }

  def copy() = new TakeAggregator(t, n)
}

class TakeByAggregator[T](var t: Type, var f: (Any) => Any, var n: Int)(implicit var tord: Ordering[T]) extends TypedAggregator[IndexedSeq[Any]] {
  def this() = this(null, null, 0)(null)

  def makeOrd(): Ordering[(Any, Any)] =
    if (tord != null)
      ExtendedOrdering.extendToNull(tord).toOrdering
        .on { case (e, k) => k.asInstanceOf[T] }
    else
      null

  var ord: Ordering[(Any, Any)] = makeOrd()

  // PriorityQueue is not serializable
  // https://issues.scala-lang.org/browse/SI-7568
  // fixed in Scala 2.11.0-M7
  var _state = if (ord != null)
    new mutable.PriorityQueue[(Any, Any)]()(ord)
  else
    null

  def result = _state.clone.dequeueAll.toArray[(Any, Any)].map(_._1).reverse: IndexedSeq[Any]

  def seqOp(x: Any) = {
    val cx = Annotation.copy(t, x)
    seqOp(cx, f(cx))
  }

  private def seqOp(x: Any, sortKey: Any) = {
    val p = (x, sortKey)
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
    agg2._state.foreach { case (x, p) => seqOp(x, p) }
  }

  def copy() = new TakeByAggregator(t, f, n)

  private def writeObject(oos: ObjectOutputStream) {
    oos.writeObject(t)
    oos.writeObject(f)
    oos.writeInt(n)
    oos.writeObject(tord)
    oos.writeObject(_state.toArray[(Any, Any)])
  }

  private def readObject(ois: ObjectInputStream) {
    t = ois.readObject().asInstanceOf[Type]
    f = ois.readObject().asInstanceOf[(Any) => Any]
    n = ois.readInt()
    tord = ois.readObject().asInstanceOf[Ordering[T]]
    ord = makeOrd()

    val elems = ois.readObject().asInstanceOf[Array[(Any, Any)]]
    _state = mutable.PriorityQueue[(Any, Any)](elems: _*)(ord)
  }
}
