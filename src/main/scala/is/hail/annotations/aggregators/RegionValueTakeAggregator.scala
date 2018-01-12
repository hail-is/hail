package is.hail.annotations.aggregators

import is.hail.annotations._
import is.hail.asm4s._
import is.hail.expr._
import is.hail.utils._

object RegionValueTakeBooleanAggregator {
  def stagedNew(v: Array[Code[_]], m: Array[Code[Boolean]]): Code[RegionValueTakeBooleanAggregator] = (v, m) match {
    case (Array(n), Array(mn)) =>
      mn.mux(
        Code._throw(Code.newInstance[RuntimeException, String]("Take-aggregator cannot take NA for length")),
        Code.newInstance[RegionValueTakeBooleanAggregator, Int](n.asInstanceOf[Code[Int]]))
  }
}

class RegionValueTakeBooleanAggregator(n: Int) extends RegionValueAggregator {
  private val ab = new MissingBooleanArrayBuilder()

  def seqOp(x: Boolean, missing: Boolean) {
    if (ab.length() < n)
      if (missing)
        ab.addMissing()
      else
        ab.add(x)
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadBoolean(off), missing)
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
}

object RegionValueTakeIntAggregator {
  def stagedNew(v: Array[Code[_]], m: Array[Code[Boolean]]): Code[RegionValueTakeIntAggregator] = (v, m) match {
    case (Array(n), Array(mn)) =>
      mn.mux(
        Code._throw(Code.newInstance[RuntimeException, String]("Take-aggregator cannot take NA for length")),
        Code.newInstance[RegionValueTakeIntAggregator, Int](n.asInstanceOf[Code[Int]]))
  }
}

class RegionValueTakeIntAggregator(n: Int) extends RegionValueAggregator {
  private val ab = new MissingIntArrayBuilder()

  def seqOp(x: Int, missing: Boolean) {
    if (ab.length() < n)
      if (missing)
        ab.addMissing()
      else
        ab.add(x)
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadInt(off), missing)
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
}

object RegionValueTakeLongAggregator {
  def stagedNew(v: Array[Code[_]], m: Array[Code[Boolean]]): Code[RegionValueTakeLongAggregator] = (v, m) match {
    case (Array(n), Array(mn)) =>
      mn.mux(
        Code._throw(Code.newInstance[RuntimeException, String]("Take-aggregator cannot take NA for length")),
        Code.newInstance[RegionValueTakeLongAggregator, Int](n.asInstanceOf[Code[Int]]))
  }
}

class RegionValueTakeLongAggregator(n: Int) extends RegionValueAggregator {
  private val ab = new MissingLongArrayBuilder()

  def seqOp(x: Long, missing: Boolean) {
    if (ab.length() < n)
      if (missing)
        ab.addMissing()
      else
        ab.add(x)
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadLong(off), missing)
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
}

object RegionValueTakeFloatAggregator {
  def stagedNew(v: Array[Code[_]], m: Array[Code[Boolean]]): Code[RegionValueTakeFloatAggregator] = (v, m) match {
    case (Array(n), Array(mn)) =>
      mn.mux(
        Code._throw(Code.newInstance[RuntimeException, String]("Take-aggregator cannot take NA for length")),
        Code.newInstance[RegionValueTakeFloatAggregator, Int](n.asInstanceOf[Code[Int]]))
  }
}

class RegionValueTakeFloatAggregator(n: Int) extends RegionValueAggregator {
  private val ab = new MissingFloatArrayBuilder()

  def seqOp(x: Float, missing: Boolean) {
    if (ab.length() < n)
      if (missing)
        ab.addMissing()
      else
        ab.add(x)
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadFloat(off), missing)
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
}

object RegionValueTakeDoubleAggregator {
  def stagedNew(v: Array[Code[_]], m: Array[Code[Boolean]]): Code[RegionValueTakeDoubleAggregator] = (v, m) match {
    case (Array(n), Array(mn)) =>
      mn.mux(
        Code._throw(Code.newInstance[RuntimeException, String]("Take-aggregator cannot take NA for length")),
        Code.newInstance[RegionValueTakeDoubleAggregator, Int](n.asInstanceOf[Code[Int]]))
  }
}

class RegionValueTakeDoubleAggregator(n: Int) extends RegionValueAggregator {
  private val ab = new MissingDoubleArrayBuilder()

  def seqOp(x: Double, missing: Boolean) {
    if (ab.length() < n)
      if (missing)
        ab.addMissing()
      else
        ab.add(x)
  }

  def seqOp(region: Region, off: Long, missing: Boolean) {
    seqOp(region.loadDouble(off), missing)
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
}
