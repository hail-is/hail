package is.hail.utils

import is.hail.expr.types._
import is.hail.annotations._
import scala.collection.mutable

class MissingBooleanArrayBuilder extends Serializable {
  private var len = 0
  private var elements = new mutable.BitSet()
  private var isMissing = new mutable.BitSet()

  def addMissing() {
    isMissing.add(len)
    len += 1
  }

  def add(x: Boolean) {
    if (x)
      elements.add(len)
    len += 1
  }

  def length(): Int = len

  def foreach(whenMissing: (Int) => Unit)(whenPresent: (Int, Boolean) => Unit) {
    var i = 0
    while (i < len) {
      if (isMissing(i))
        whenMissing(i)
      else
        whenPresent(i, elements(i))
      i += 1
    }
  }

  val typ = TArray(TBoolean())

  def write(rvb: RegionValueBuilder) {
    rvb.startArray(len)
    var i = 0
    while (i < len) {
      if (isMissing(i))
        rvb.setMissing()
      else
        rvb.addBoolean(elements(i))
      i += 1
    }
    rvb.endArray()
  }

  def clear() {
    len = 0
    elements.clear()
    isMissing.clear()
  }

  override def clone(): MissingBooleanArrayBuilder = {
    val ab = new MissingBooleanArrayBuilder()
    ab.len = len
    ab.elements = elements.clone()
    ab.isMissing = isMissing.clone()
    ab
  }
}
