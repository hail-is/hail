package is.hail.annotations.aggregators

import is.hail.annotations.{Annotation, Region, RegionValueBuilder, SafeRow}
import is.hail.expr.types.Type
import is.hail.utils._

import scala.collection.mutable

class RegionValueCounterBooleanAggregator extends RegionValueAggregator {
  private val a = Array(0L, 0L, 0L)

  def seqOp(region: Region, x: Boolean, missing: Boolean) {
    if (missing)
      a(2) += 1
    else if (x)
      a(1) += 1
    else
      a(0) += 1
  }

  override def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCounterBooleanAggregator]
    a(0) += other.a(0)
    a(1) += other.a(1)
    a(2) += other.a(2)
  }

  override def result(rvb: RegionValueBuilder) {
    rvb.startArray(a.count(_ != 0))

    if (a(0) > 0) {
      rvb.startStruct()
      rvb.addBoolean(false)
      rvb.addLong(a(0))
      rvb.endStruct()
    }

    if (a(1) > 0) {
      rvb.startStruct()
      rvb.addBoolean(true)
      rvb.addLong(a(1))
      rvb.endStruct()
    }

    if (a(2) > 0) {
      rvb.startStruct()
      rvb.setMissing()
      rvb.addLong(a(2))
      rvb.endStruct()
    }

    rvb.endArray()
  }

  override def copy(): RegionValueCounterBooleanAggregator = new RegionValueCounterBooleanAggregator()

  override def clear() {
    a(0) = 0L
    a(1) = 0L
    a(2) = 0L
  }
}

class RegionValueCounterIntAggregator extends RegionValueAggregator {
  private val m = mutable.Map[Int, Long]()
  private var nMissing = 0L

  def seqOp(region: Region, x: Int, missing: Boolean) {
    if (missing)
      nMissing += 1
    else
      m.updateValue(x, 0L, _ + 1L)
  }

  override def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCounterIntAggregator]
    other.m.foreach { case (k, v) =>
      other.m.updateValue(k, 0L, _ + v)
    }

    nMissing += other.nMissing
  }

  override def result(rvb: RegionValueBuilder) {
    rvb.startArray(m.size + (if (nMissing == 0L) 0 else 1))
    m.foreach { case (k, v) =>
      rvb.startStruct()
      rvb.addInt(k)
      rvb.addLong(v)
      rvb.endStruct()
    }

    if (nMissing > 0L) {
      rvb.startStruct()
      rvb.setMissing()
      rvb.addLong(nMissing)
      rvb.endStruct()
    }

    rvb.endArray()
  }

  override def copy(): RegionValueCounterIntAggregator = new RegionValueCounterIntAggregator()

  override def clear() {
    m.clear()
  }
}

class RegionValueCounterLongAggregator extends RegionValueAggregator {
  private val m = mutable.Map[Long, Long]()
  private var nMissing = 0L

  def seqOp(region: Region, x: Long, missing: Boolean) {
    if (missing)
      nMissing += 1
    else
      m.updateValue(x, 0L, _ + 1L)
  }

  override def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCounterLongAggregator]
    other.m.foreach { case (k, v) =>
      other.m.updateValue(k, 0L, _ + v)
    }

    nMissing += other.nMissing
  }

  override def result(rvb: RegionValueBuilder) {
    rvb.startArray(m.size + (if (nMissing == 0L) 0 else 1))
    m.foreach { case (k, v) =>
      rvb.startStruct()
      rvb.addLong(k)
      rvb.addLong(v)
      rvb.endStruct()
    }

    if (nMissing > 0L) {
      rvb.startStruct()
      rvb.setMissing()
      rvb.addLong(nMissing)
      rvb.endStruct()
    }

    rvb.endArray()
  }

  override def copy(): RegionValueCounterLongAggregator = new RegionValueCounterLongAggregator()

  override def clear() {
    m.clear()
  }
}

class RegionValueCounterFloatAggregator extends RegionValueAggregator {
  private val m = mutable.Map[Float, Long]()
  private var nMissing = 0L

  def seqOp(region: Region, x: Float, missing: Boolean) {
    if (missing)
      nMissing += 1
    else
      m.updateValue(x, 0L, _ + 1L)
  }

  override def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCounterFloatAggregator]
    other.m.foreach { case (k, v) =>
      other.m.updateValue(k, 0L, _ + v)
    }

    nMissing += other.nMissing
  }

  override def result(rvb: RegionValueBuilder) {
    rvb.startArray(m.size + (if (nMissing == 0L) 0 else 1))
    m.foreach { case (k, v) =>
      rvb.startStruct()
      rvb.addFloat(k)
      rvb.addLong(v)
      rvb.endStruct()
    }

    if (nMissing > 0L) {
      rvb.startStruct()
      rvb.setMissing()
      rvb.addLong(nMissing)
      rvb.endStruct()
    }

    rvb.endArray()
  }

  override def copy(): RegionValueCounterFloatAggregator = new RegionValueCounterFloatAggregator()

  override def clear() {
    m.clear()
  }
}

class RegionValueCounterDoubleAggregator extends RegionValueAggregator {
  private val m = mutable.Map[Double, Long]()
  private var nMissing = 0L

  def seqOp(region: Region, x: Double, missing: Boolean) {
    if (missing)
      nMissing += 1
    else
      m.updateValue(x, 0L, _ + 1L)
  }

  override def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCounterDoubleAggregator]
    other.m.foreach { case (k, v) =>
      other.m.updateValue(k, 0L, _ + v)
    }

    nMissing += other.nMissing
  }

  override def result(rvb: RegionValueBuilder) {
    rvb.startArray(m.size + (if (nMissing == 0L) 0 else 1))
    m.foreach { case (k, v) =>
      rvb.startStruct()
      rvb.addDouble(k)
      rvb.addLong(v)
      rvb.endStruct()
    }

    if (nMissing > 0L) {
      rvb.startStruct()
      rvb.setMissing()
      rvb.addLong(nMissing)
      rvb.endStruct()
    }

    rvb.endArray()
  }

  override def copy(): RegionValueCounterDoubleAggregator = new RegionValueCounterDoubleAggregator()

  override def clear() {
    m.clear()
  }
}

class RegionValueCounterAnnotationAggregator(t: Type) extends RegionValueAggregator {
  private val m = mutable.Map[Annotation, Long]()

  def seqOp(region: Region, offset: Long, missing: Boolean) {
    val a = if (missing) null else SafeRow.read(t, region, offset)
    m.updateValue(a, 0L, _ + 1L)
  }

  override def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCounterAnnotationAggregator]
    other.m.foreach { case (k, v) =>
      m.updateValue(k, 0L, _ + v)
    }
  }

  override def result(rvb: RegionValueBuilder) {
    rvb.startArray(m.size)
    m.foreach { case (k, v) =>
      rvb.startStruct()
      if (k == null)
        rvb.setMissing()
      else
        rvb.addAnnotation(t, k)
      rvb.addLong(v)
      rvb.endStruct()
    }
    rvb.endArray()
  }

  override def copy(): RegionValueCounterAnnotationAggregator = new RegionValueCounterAnnotationAggregator(t)

  override def clear() {
    m.clear()
  }
}
