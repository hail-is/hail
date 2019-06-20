package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.expr.types.physical.PType
import is.hail.expr.types.virtual.Type
import is.hail.utils._

class RegionValueTakeBooleanAggregator(n: Int) extends RegionValueAggregator {
  private var ab = new MissingBooleanArrayBuilder()

  def seqOp(region: Region, x: Boolean, missing: Boolean) {
    if (ab.length() < n)
      if (missing)
        ab.addMissing()
      else
        ab.add(x)
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueTakeBooleanAggregator]
    that.ab.foreach { _ =>
      if (ab.length() < n)
        ab.addMissing()
    } { (_, v) =>
      if (ab.length() < n)
        ab.add(v)
    }
  }

  def result(rvb: RegionValueBuilder) {
    ab.write(rvb)
  }

  def newInstance(): RegionValueTakeBooleanAggregator = new RegionValueTakeBooleanAggregator(n)

  def copy(): RegionValueTakeBooleanAggregator = {
    val rva = new RegionValueTakeBooleanAggregator(n)
    rva.ab = ab.clone()
    rva
  }

  def clear() {
    ab.clear()
  }
}

class RegionValueTakeIntAggregator(n: Int) extends RegionValueAggregator {
  private var ab = new MissingIntArrayBuilder()

  def seqOp(region: Region, x: Int, missing: Boolean) {
    if (ab.length() < n)
      if (missing)
        ab.addMissing()
      else
        ab.add(x)
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueTakeIntAggregator]
    that.ab.foreach { _ =>
      if (ab.length() < n)
        ab.addMissing()
    } { (_, v) =>
      if (ab.length() < n)
        ab.add(v)
    }
  }

  def result(rvb: RegionValueBuilder) {
    ab.write(rvb)
  }

  def newInstance(): RegionValueTakeIntAggregator = new RegionValueTakeIntAggregator(n)

  def copy(): RegionValueTakeIntAggregator = {
    val rva = new RegionValueTakeIntAggregator(n)
    rva.ab = ab.clone()
    rva
  }

  def clear() {
    ab.clear()
  }
}

class RegionValueTakeLongAggregator(n: Int) extends RegionValueAggregator {
  private var ab = new MissingLongArrayBuilder()

  def seqOp(region: Region, x: Long, missing: Boolean) {
    if (ab.length() < n)
      if (missing)
        ab.addMissing()
      else
        ab.add(x)
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueTakeLongAggregator]
    that.ab.foreach { _ =>
      if (ab.length() < n)
        ab.addMissing()
    } { (_, v) =>
      if (ab.length() < n)
        ab.add(v)
    }
  }

  def result(rvb: RegionValueBuilder) {
    ab.write(rvb)
  }

  def newInstance(): RegionValueTakeLongAggregator = new RegionValueTakeLongAggregator(n)

  def copy(): RegionValueTakeLongAggregator = {
    val rva = new RegionValueTakeLongAggregator(n)
    rva.ab = ab.clone()
    rva
  }

  def clear() {
    ab.clear()
  }
}

class RegionValueTakeFloatAggregator(n: Int) extends RegionValueAggregator {
  private var ab = new MissingFloatArrayBuilder()

  def seqOp(region: Region, x: Float, missing: Boolean) {
    if (ab.length() < n)
      if (missing)
        ab.addMissing()
      else
        ab.add(x)
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueTakeFloatAggregator]
    that.ab.foreach { _ =>
      if (ab.length() < n)
        ab.addMissing()
    } { (_, v) =>
      if (ab.length() < n)
        ab.add(v)
    }
  }

  def result(rvb: RegionValueBuilder) {
    ab.write(rvb)
  }

  def newInstance(): RegionValueTakeFloatAggregator = new RegionValueTakeFloatAggregator(n)

  def copy(): RegionValueTakeFloatAggregator = {
    val rva = new RegionValueTakeFloatAggregator(n)
    rva.ab = ab.clone()
    rva
  }

  def clear() {
    ab.clear()
  }
}

class RegionValueTakeDoubleAggregator(n: Int) extends RegionValueAggregator {
  private var ab = new MissingDoubleArrayBuilder()

  def seqOp(region: Region, x: Double, missing: Boolean) {
    if (ab.length() < n)
      if (missing)
        ab.addMissing()
      else
        ab.add(x)
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueTakeDoubleAggregator]
    that.ab.foreach { _ =>
      if (ab.length() < n)
        ab.addMissing()
    } { (_, v) =>
      if (ab.length() < n)
        ab.add(v)
    }
  }

  def result(rvb: RegionValueBuilder) {
    ab.write(rvb)
  }

  def newInstance(): RegionValueTakeDoubleAggregator = new RegionValueTakeDoubleAggregator(n)

  def copy(): RegionValueTakeDoubleAggregator = {
    val rva = new RegionValueTakeDoubleAggregator(n)
    rva.ab = ab.clone()
    rva
  }

  def clear() {
    ab.clear()
  }
}

// FIXME make it a ptype
class RegionValueTakeAnnotationAggregator(n: Int, ptype: PType) extends RegionValueAggregator {
  private var ab = new MissingAnnotationArrayBuilder()

  def seqOp(region: Region, offset: Long, missing: Boolean) {
    if (ab.length() < n)
      if (missing)
        ab.addMissing()
      else
        ab.add(SafeRow.read(ptype, region, offset))
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValueTakeAnnotationAggregator]
    that.ab.foreach { _ =>
      if (ab.length() < n)
        ab.addMissing()
    } { (_, v) =>
      if (ab.length() < n)
        ab.add(v)
    }
  }

  def result(rvb: RegionValueBuilder) {
    ab.write(rvb, ptype)
  }

  def newInstance(): RegionValueTakeAnnotationAggregator = new RegionValueTakeAnnotationAggregator(n, ptype)

  def copy(): RegionValueTakeAnnotationAggregator = {
    val rva = new RegionValueTakeAnnotationAggregator(n, ptype)
    rva.ab = ab.clone()
    rva
  }

  def clear() {
    ab.clear()
  }
}
