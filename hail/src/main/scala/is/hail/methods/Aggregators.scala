package is.hail.methods

import is.hail.annotations._
import is.hail.expr._
import is.hail.expr.types.physical.PType
import is.hail.expr.types.virtual.Type
import is.hail.stats._
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.sql.Row

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.language.implicitConversions
import scala.reflect.ClassTag

object Aggregators {

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
            s"""cannot aggregate arrays of unequal length with 'sum'
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
          s"""cannot aggregate arrays of unequal length with 'sum'
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

  var _state = if (ord != null)
    new mutable.PriorityQueue[(Any, Any)]()(ord)
  else
    null

  def result = _state.clone.dequeueAll.toArray[(Any, Any)].map(_._1).reverse: IndexedSeq[Any]

  def seqOp(x: Any) = {
    val cx = Annotation.copy(t, x)
    seqOp(cx, f(cx))
  }

  def seqOp(x: Any, sortKey: Any) = {
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
}

class LinearRegressionAggregator(xF: (Any) => Any, k: Int, k0: Int, xType: Type) extends TypedAggregator[Any] {
  var combiner = new LinearRegressionCombiner(k, k0, xType.physicalType)

  def seqOp(a: Any) = {
    if (a != null) {
      val y = a.asInstanceOf[Double]
      val x = xF(y)
      if (x != null)
        combiner.merge(y, x.asInstanceOf[Array[Double]])
    }
  }

  def seqOp(y: Any, x: Any) {
    if (y != null && x != null) {
      combiner.merge(y.asInstanceOf[Double], x.asInstanceOf[IndexedSeq[Double]])
    }
  }

  def combOp(agg2: this.type) {
    combiner.merge(agg2.combiner)
  }

  def result: Annotation = combiner.result()

  def copy() = {
    val lra = new LinearRegressionAggregator(xF, k, k0, xType)
    lra.combiner = combiner.copy()
    lra
  }
}

class KeyedAggregator[T, K](aggregator: TypedAggregator[T]) extends TypedAggregator[Map[Any, T]] {
  private val m = new java.util.HashMap[Any, TypedAggregator[T]]()

  def result = m.asScala.map { case (k, v) => (k, v.result) }.toMap

  def seqOp(x: Any) {
    val cx = x.asInstanceOf[Row]
    if (cx != null)
      seqOp(cx.get(0), cx.get(1))
    else
      seqOp(null, null)
  }

  private def seqOp(key: Any, x: Any) {
    var agg = m.get(key)
    if (agg == null) {
      agg = aggregator.copy()
      m.put(key, agg)
    }
    val r = x.asInstanceOf[Row]
    agg match {
      case tagg: KeyedAggregator[_, _] => agg.seqOp(x)
      case tagg: CountAggregator => agg.seqOp(0)
      case tagg: TakeByAggregator[_] =>
        agg.asInstanceOf[TakeByAggregator[_]].seqOp(r.get(0), r.get(1))
      case _ => agg.seqOp(r.get(0))
    }
  }

  def combOp(agg2: this.type) {
    agg2.m.asScala.foreach { case (k, v2) =>
      val agg = m.get(k)
      if (agg == null)
        m.put(k, v2)
      else {
        agg.combOp(v2.asInstanceOf[agg.type])
      }
    }
  }

  def copy() = new KeyedAggregator(aggregator.copy())
}

class DownsampleAggregator(nDivisions: Int, getYL: Any => (Any, Any)) extends TypedAggregator[IndexedSeq[Row]] {
  require(nDivisions > 0)

  var _state = new DownsampleCombiner(nDivisions)

  def result: IndexedSeq[Row] = _state.toRes

  def seqOp(x: Any, y: Any, l: Any) = {
    if (x != null && y != null) {
      val labelArgs = l.asInstanceOf[IndexedSeq[String]]
      _state.merge(x.asInstanceOf[Double], y.asInstanceOf[Double], labelArgs)
    }
  }

  def seqOp(x: Any) = {
    if (x != null) {
      val y = getYL(x)._1
      val l = getYL(x)._2
      val labelArgs = l.asInstanceOf[IndexedSeq[String]]
      if (y != null)
        _state.merge(x.asInstanceOf[Double], y.asInstanceOf[Double], labelArgs)
    }
  }

  def combOp(agg2: this.type) = _state.merge(agg2._state)

  def copy() = new DownsampleAggregator(nDivisions, getYL)
}
