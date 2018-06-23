package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr._
import is.hail.expr.types.Type
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

  def copy(): RegionValueTakeBooleanAggregator = new RegionValueTakeBooleanAggregator(n)

  def deepCopy(): RegionValueTakeBooleanAggregator = {
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

  def copy(): RegionValueTakeIntAggregator = new RegionValueTakeIntAggregator(n)

  def deepCopy(): RegionValueTakeIntAggregator = {
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

  def copy(): RegionValueTakeLongAggregator = new RegionValueTakeLongAggregator(n)

  def deepCopy(): RegionValueTakeLongAggregator = {
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

  def copy(): RegionValueTakeFloatAggregator = new RegionValueTakeFloatAggregator(n)

  def deepCopy(): RegionValueTakeFloatAggregator = {
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

  def copy(): RegionValueTakeDoubleAggregator = new RegionValueTakeDoubleAggregator(n)

  def deepCopy(): RegionValueTakeDoubleAggregator = {
    val rva = new RegionValueTakeDoubleAggregator(n)
    rva.ab = ab.clone()
    rva
  }

  def clear() {
    ab.clear()
  }
}

class RegionValueTakeAnnotationAggregator(n: Int, t: Type) extends RegionValueAggregator {
  private var ab = new MissingAnnotationArrayBuilder()

  def seqOp(region: Region, offset: Long, missing: Boolean) {
    if (ab.length() < n)
      if (missing)
        ab.addMissing()
      else
        ab.add(SafeRow.read(t, region, offset))
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
    ab.write(rvb, t)
  }

  def copy(): RegionValueTakeAnnotationAggregator = new RegionValueTakeAnnotationAggregator(n, t)

  def deepCopy(): RegionValueTakeAnnotationAggregator = {
    val rva = new RegionValueTakeAnnotationAggregator(n, t)
    rva.ab = ab.clone()
    rva
  }

  def clear() {
    ab.clear()
  }
}
