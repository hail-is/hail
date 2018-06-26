package is.hail.annotations.aggregators

import is.hail.annotations.{Annotation, Region, RegionValueBuilder, SafeRow}
import is.hail.expr.types.Type
import is.hail.utils._

import scala.collection.mutable

class RegionValueCounterAggregator(t: Type) extends RegionValueAggregator {
  var m = mutable.Map[Any, Long]()

  def seqOp(a: Any) {
    m.updateValue(a, 0L, _ + 1L)
  }

  override def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCounterAggregator]
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

  override def newInstance(): RegionValueCounterAggregator = new RegionValueCounterAggregator(t)

  override def copy(): RegionValueCounterAggregator = {
    val rva = new RegionValueCounterAggregator(t)
    rva.m = m.clone()
    rva
  }

  override def clear() {
    m.clear()
  }
}

class RegionValueCounterBooleanAggregator extends RegionValueAggregator {
  private var nTrue = 0L
  private var nFalse = 0L
  private var nMissing = 0L

  def seqOp(region: Region, x: Boolean, missing: Boolean) {
    if (missing)
      nMissing += 1
    else if (x)
      nTrue += 1
    else
      nFalse += 1
  }

  override def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCounterBooleanAggregator]
    nTrue += other.nTrue
    nFalse += other.nFalse
    nMissing += other.nMissing
  }

  override def result(rvb: RegionValueBuilder) {
    rvb.startArray((nMissing != 0).toInt + (nTrue != 0).toInt + (nFalse != 0).toInt)

    if (nFalse > 0) {
      rvb.startStruct()
      rvb.addBoolean(false)
      rvb.addLong(nFalse)
      rvb.endStruct()
    }

    if (nTrue > 0) {
      rvb.startStruct()
      rvb.addBoolean(true)
      rvb.addLong(nTrue)
      rvb.endStruct()
    }

    if (nMissing > 0) {
      rvb.startStruct()
      rvb.setMissing()
      rvb.addLong(nMissing)
      rvb.endStruct()
    }

    rvb.endArray()
  }

  override def newInstance(): RegionValueCounterBooleanAggregator = new RegionValueCounterBooleanAggregator()

  override def copy(): RegionValueCounterBooleanAggregator = {
    val rva = new RegionValueCounterBooleanAggregator()
    rva.nTrue = nTrue
    rva.nFalse = nFalse
    rva.nMissing = nMissing
    rva
  }

  override def clear() {
    nTrue = 0L
    nFalse = 0L
    nMissing = 0L
  }
}

class RegionValueCounterIntAggregator(t: Type) extends RegionValueCounterAggregator(t) {
  def seqOp(region: Region, x: Int, missing: Boolean) {
    seqOp(if (missing) null else x)
  }

  override def newInstance(): RegionValueCounterIntAggregator = new RegionValueCounterIntAggregator(t)

  override def copy(): RegionValueCounterIntAggregator = {
    val rva = new RegionValueCounterIntAggregator(t)
    rva.m = m.clone()
    rva
  }
}

class RegionValueCounterLongAggregator(t: Type) extends RegionValueCounterAggregator(t) {
  def seqOp(region: Region, x: Long, missing: Boolean) {
    seqOp(if (missing) null else x)
  }

  override def newInstance(): RegionValueCounterLongAggregator = new RegionValueCounterLongAggregator(t)

  override def copy(): RegionValueCounterLongAggregator = {
    val rva = new RegionValueCounterLongAggregator(t)
    rva.m = m.clone()
    rva
  }
}

class RegionValueCounterFloatAggregator(t: Type) extends RegionValueCounterAggregator(t) {
  def seqOp(region: Region, x: Float, missing: Boolean) {
    seqOp(if (missing) null else x)
  }

  override def newInstance(): RegionValueCounterFloatAggregator = new RegionValueCounterFloatAggregator(t)

  override def copy(): RegionValueCounterFloatAggregator = {
    val rva = new RegionValueCounterFloatAggregator(t)
    rva.m = m.clone()
    rva
  }
}

class RegionValueCounterDoubleAggregator(t: Type) extends RegionValueCounterAggregator(t) {
  def seqOp(region: Region, x: Double, missing: Boolean) {
    seqOp(if (missing) null else x)
  }

  override def newInstance(): RegionValueCounterDoubleAggregator = new RegionValueCounterDoubleAggregator(t)

  override def copy(): RegionValueCounterDoubleAggregator = {
    val rva = new RegionValueCounterDoubleAggregator(t)
    rva.m = m.clone()
    rva
  }
}

class RegionValueCounterAnnotationAggregator(t: Type) extends RegionValueCounterAggregator(t) {
  def seqOp(region: Region, offset: Long, missing: Boolean) {
    seqOp(if (missing) null else SafeRow.read(t, region, offset))
  }

  override def newInstance(): RegionValueCounterAnnotationAggregator = new RegionValueCounterAnnotationAggregator(t)

  override def copy(): RegionValueCounterAnnotationAggregator = {
    val rva = new RegionValueCounterAnnotationAggregator(t)
    rva.m = m.clone()
    rva
  }
}