package is.hail.annotations.aggregators

import is.hail.annotations.{Region, RegionValueBuilder, SafeRow}
import is.hail.expr.types.{TSet, Type}

import scala.collection.mutable

class RegionValueCollectAsSetBooleanAggregator extends RegionValueAggregator {
  private var hasTrue = false
  private var hasFalse = false
  private var hasMissing = false

  def seqOp(region: Region, b: Boolean, missing: Boolean) {
    if (missing)
      hasMissing = true
    else {
      if (b)
        hasTrue = true
      else
        hasFalse = true
    }
  }

  def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCollectAsSetBooleanAggregator]
    hasTrue |= other.hasTrue
    hasFalse |= other.hasFalse
    hasMissing |= other.hasMissing
  }

  def result(rvb: RegionValueBuilder) {
    rvb.startArray((if (hasTrue) 1 else 0) + (if (hasFalse) 1 else 0) + (if (hasMissing) 1 else 0))
    if (hasTrue)
      rvb.addBoolean(true)
    if (hasFalse)
      rvb.addBoolean(false)
    if (hasMissing)
      rvb.setMissing()
    rvb.endArray()
  }

  def newInstance(): RegionValueCollectAsSetBooleanAggregator = new RegionValueCollectAsSetBooleanAggregator()

  override def copy(): RegionValueCollectAsSetBooleanAggregator = {
    val rva = new RegionValueCollectAsSetBooleanAggregator()
    rva.hasTrue = hasTrue
    rva.hasFalse = hasFalse
    rva.hasMissing = hasMissing
    rva
  }

  def clear() {
    hasTrue = false
    hasFalse = false
    hasMissing = false
  }
}

class RegionValueCollectAsSetIntAggregator extends RegionValueAggregator {
  private var values = mutable.Set[Int]()
  private var hasMissing = false

  def seqOp(region: Region, x: Int, missing: Boolean) {
    if (missing)
      hasMissing = true
    else
      values.add(x)
  }

  def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCollectAsSetIntAggregator]
    hasMissing |= other.hasMissing
    values ++= other.values
  }

  def result(rvb: RegionValueBuilder) {
    rvb.startArray(if (hasMissing) values.size + 1 else values.size)
    values.toArray.sorted.foreach { i =>
      rvb.addInt(i)
    }
    if (hasMissing)
      rvb.setMissing()
    rvb.endArray()
  }

  def newInstance(): RegionValueCollectAsSetIntAggregator = new RegionValueCollectAsSetIntAggregator()

  override def copy(): RegionValueCollectAsSetIntAggregator = {
    val rva = new RegionValueCollectAsSetIntAggregator()
    rva.values = values.clone()
    rva
  }

  def clear() {
    values.clear()
    hasMissing = false
  }
}

class RegionValueCollectAsSetLongAggregator extends RegionValueAggregator {
  private var values = mutable.Set[Long]()
  private var hasMissing = false

  def seqOp(region: Region, x: Long, missing: Boolean) {
    if (missing)
      hasMissing = true
    else
      values.add(x)
  }

  def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCollectAsSetLongAggregator]
    hasMissing |= other.hasMissing
    values ++= other.values
  }

  def result(rvb: RegionValueBuilder) {
    rvb.startArray(if (hasMissing) values.size + 1 else values.size)
    values.toArray.sorted.foreach { i =>
      rvb.addLong(i)
    }
    if (hasMissing)
      rvb.setMissing()
    rvb.endArray()
  }

  def newInstance(): RegionValueCollectAsSetLongAggregator = new RegionValueCollectAsSetLongAggregator()

  override def copy(): RegionValueCollectAsSetLongAggregator = {
    val rva = new RegionValueCollectAsSetLongAggregator()
    rva.values = values.clone()
    rva
  }

  def clear() {
    values.clear()
    hasMissing = false
  }
}

class RegionValueCollectAsSetFloatAggregator extends RegionValueAggregator {
  private var values = mutable.Set[Float]()
  private var hasMissing = false

  def seqOp(region: Region, x: Float, missing: Boolean) {
    if (missing)
      hasMissing = true
    else
      values.add(x)
  }

  def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCollectAsSetFloatAggregator]
    hasMissing |= other.hasMissing
    values ++= other.values
  }

  def result(rvb: RegionValueBuilder) {
    rvb.startArray(if (hasMissing) values.size + 1 else values.size)
    values.toArray.sorted.foreach { i =>
      rvb.addFloat(i)
    }
    if (hasMissing)
      rvb.setMissing()
    rvb.endArray()
  }

  def newInstance(): RegionValueCollectAsSetFloatAggregator = new RegionValueCollectAsSetFloatAggregator()

  override def copy(): RegionValueCollectAsSetFloatAggregator = {
    val rva = new RegionValueCollectAsSetFloatAggregator()
    rva.values = values.clone()
    rva
  }

  def clear() {
    values.clear()
    hasMissing = false
  }
}

class RegionValueCollectAsSetDoubleAggregator extends RegionValueAggregator {
  private var values = mutable.Set[Double]()
  private var hasMissing = false

  def seqOp(region: Region, x: Double, missing: Boolean) {
    if (missing)
      hasMissing = true
    else
      values.add(x)
  }

  def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCollectAsSetDoubleAggregator]
    hasMissing |= other.hasMissing
    values ++= other.values
  }

  def result(rvb: RegionValueBuilder) {
    rvb.startArray(if (hasMissing) values.size + 1 else values.size)
    values.toArray.sorted.foreach { i =>
      rvb.addDouble(i)
    }
    if (hasMissing)
      rvb.setMissing()
    rvb.endArray()
  }

  def newInstance(): RegionValueCollectAsSetDoubleAggregator = new RegionValueCollectAsSetDoubleAggregator()

  override def copy(): RegionValueCollectAsSetDoubleAggregator = {
    val rva = new RegionValueCollectAsSetDoubleAggregator()
    rva.values = values.clone()
    rva
  }

  def clear() {
    values.clear()
    hasMissing = false
  }
}

class RegionValueCollectAsSetAnnotationAggregator(val typ: Type) extends RegionValueAggregator {
  private var values = mutable.Set[Any]()

  def seqOp(region: Region, offset: Long, missing: Boolean) {
    if (missing)
      values.add(null)
    else
      values.add(SafeRow.read(typ, region, offset))
  }

  def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCollectAsSetAnnotationAggregator]
    values ++= other.values
  }

  def result(rvb: RegionValueBuilder) {
    rvb.addAnnotation(TSet(typ), values.toSet)
  }

  def newInstance(): RegionValueCollectAsSetAnnotationAggregator = new RegionValueCollectAsSetAnnotationAggregator(typ)

  override def copy(): RegionValueCollectAsSetAnnotationAggregator = {
    val rva = new RegionValueCollectAsSetAnnotationAggregator(typ)
    rva.values = values.clone()
    rva
  }

  def clear() {
    values.clear()
  }
}

