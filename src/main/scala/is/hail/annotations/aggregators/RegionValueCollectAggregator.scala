package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.expr.types.{TBaseStruct, Type}
import is.hail.utils._

class RegionValueCollectBooleanAggregator extends RegionValueAggregator {
  private var ab = new MissingBooleanArrayBuilder()

  def seqOp(region: Region, b: Boolean, missing: Boolean) {
    if (missing)
      ab.addMissing()
    else
      ab.add(b)
  }

  def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCollectBooleanAggregator]
    other.ab.foreach { _ =>
      ab.addMissing()
    } { (_, x) =>
      ab.add(x)
    }
  }

  def result(rvb: RegionValueBuilder) {
    ab.write(rvb)
  }

  def newInstance(): RegionValueCollectBooleanAggregator = new RegionValueCollectBooleanAggregator()

  def copy(): RegionValueCollectBooleanAggregator = {
    val rva = new RegionValueCollectBooleanAggregator()
    rva.ab = ab.clone()
    rva
  }

  def clear() {
    ab.clear()
  }
}

class RegionValueCollectIntAggregator extends RegionValueAggregator {
  private var ab = new MissingIntArrayBuilder()

  def seqOp(region: Region, x: Int, missing: Boolean) {
    if (missing)
      ab.addMissing()
    else
      ab.add(x)
  }

  def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCollectIntAggregator]
    other.ab.foreach { _ =>
      ab.addMissing()
    } { (_, x) =>
      ab.add(x)
    }
  }

  def result(rvb: RegionValueBuilder) {
    ab.write(rvb)
  }

  def newInstance(): RegionValueCollectIntAggregator = new RegionValueCollectIntAggregator()

  def copy(): RegionValueCollectIntAggregator = {
    val rva = new RegionValueCollectIntAggregator()
    rva.ab = ab.clone()
    rva
  }

  def clear() {
    ab.clear()
  }
}

class RegionValueCollectLongAggregator extends RegionValueAggregator {
  private var ab = new MissingLongArrayBuilder()

  def seqOp(region: Region, x: Long, missing: Boolean) {
    if (missing)
      ab.addMissing()
    else
      ab.add(x)
  }

  def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCollectLongAggregator]
    other.ab.foreach { _ =>
      ab.addMissing()
    } { (_, x) =>
      ab.add(x)
    }
  }

  def result(rvb: RegionValueBuilder) {
    ab.write(rvb)
  }

  def newInstance(): RegionValueCollectLongAggregator = new RegionValueCollectLongAggregator()

  def copy(): RegionValueCollectLongAggregator = {
    val rva = new RegionValueCollectLongAggregator()
    rva.ab = ab.clone()
    rva
  }

  def clear() {
    ab.clear()
  }
}

class RegionValueCollectFloatAggregator extends RegionValueAggregator {
  private var ab = new MissingFloatArrayBuilder()

  def seqOp(region: Region, x: Float, missing: Boolean) {
    if (missing)
      ab.addMissing()
    else
      ab.add(x)
  }

  def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCollectFloatAggregator]
    other.ab.foreach { _ =>
      ab.addMissing()
    } { (_, x) =>
      ab.add(x)
    }
  }

  def result(rvb: RegionValueBuilder) {
    ab.write(rvb)
  }

  def newInstance(): RegionValueCollectFloatAggregator = new RegionValueCollectFloatAggregator()

  def copy(): RegionValueCollectFloatAggregator = {
    val rva = new RegionValueCollectFloatAggregator()
    rva.ab = ab.clone()
    rva
  }

  def clear() {
    ab.clear()
  }
}

class RegionValueCollectDoubleAggregator extends RegionValueAggregator {
  private var ab = new MissingDoubleArrayBuilder()

  def seqOp(region: Region, x: Double, missing: Boolean) {
    if (missing)
      ab.addMissing()
    else
      ab.add(x)
  }

  def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCollectDoubleAggregator]
    other.ab.foreach { _ =>
      ab.addMissing()
    } { (_, x) =>
      ab.add(x)
    }
  }

  def result(rvb: RegionValueBuilder) {
    ab.write(rvb)
  }

  def newInstance(): RegionValueCollectDoubleAggregator = new RegionValueCollectDoubleAggregator()

  def copy(): RegionValueCollectDoubleAggregator = {
    val rva = new RegionValueCollectDoubleAggregator()
    rva.ab = ab.clone()
    rva
  }

  def clear() {
    ab.clear()
  }
}

class RegionValueCollectAnnotationAggregator(t: Type) extends RegionValueAggregator {
  private var ab = new MissingAnnotationArrayBuilder()

  def seqOp(region: Region, offset: Long, missing: Boolean) {
    if (missing)
      ab.addMissing()
    else
      ab.add(SafeRow.read(t, region, offset))
  }

  def combOp(agg2: RegionValueAggregator) {
    val other = agg2.asInstanceOf[RegionValueCollectAnnotationAggregator]
    other.ab.foreach { _ =>
      ab.addMissing()
    } { (_, x) =>
      ab.add(x)
    }
  }

  def result(rvb: RegionValueBuilder) {
    ab.write(rvb, t)
  }

  def newInstance(): RegionValueCollectAnnotationAggregator = new RegionValueCollectAnnotationAggregator(t)

  def copy(): RegionValueCollectAnnotationAggregator = {
    val rva = new RegionValueCollectAnnotationAggregator(t)
    rva.ab = ab.clone()
    rva
  }

  def clear() {
    ab.clear()
  }
}
