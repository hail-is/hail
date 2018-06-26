package is.hail.annotations.aggregators

import is.hail.annotations.{Region, RegionValueBuilder, SafeRow}
import is.hail.expr.types.Type

import scala.collection.mutable

class RegionValueTakeByAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueAggregator {
  assert(n >= 0)

  val ord = keyType.ordering.toOrdering.on[(Any, Any)] { case (e, k) => k }
  var _state = new mutable.PriorityQueue[(Any, Any)]()(ord)

  def seqOp(p: (Any, Any)) {
    if (_state.length < n)
      _state += p
    else {
      if (ord.compare(p, _state.head) < 0) {
        _state.dequeue()
        _state += p
      }
    }
  }

  override def combOp(agg2: RegionValueAggregator) {
    agg2.asInstanceOf[RegionValueTakeByAggregator]._state.foreach(seqOp)
  }

  override def result(rvb: RegionValueBuilder) {
    rvb.startArray(_state.length)
    _state.clone.dequeueAll.toArray[(Any, Any)].map(_._1).reverse.foreach { a =>
      rvb.addAnnotation(aggType, a)
    }
    rvb.endArray()
  }

  override def newInstance(): RegionValueTakeByAggregator = new RegionValueTakeByAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByAggregator = {
    val rva = new RegionValueTakeByAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }

  override def clear() {
    _state.clear()
  }
}

class RegionValueTakeByBooleanBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByBooleanBooleanAggregator = new RegionValueTakeByBooleanBooleanAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByBooleanBooleanAggregator = {
    val rva = new RegionValueTakeByBooleanBooleanAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByBooleanIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByBooleanIntAggregator = new RegionValueTakeByBooleanIntAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByBooleanIntAggregator = {
    val rva = new RegionValueTakeByBooleanIntAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByBooleanLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByBooleanLongAggregator = new RegionValueTakeByBooleanLongAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByBooleanLongAggregator = {
    val rva = new RegionValueTakeByBooleanLongAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByBooleanFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByBooleanFloatAggregator = new RegionValueTakeByBooleanFloatAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByBooleanFloatAggregator = {
    val rva = new RegionValueTakeByBooleanFloatAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByBooleanDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByBooleanDoubleAggregator = new RegionValueTakeByBooleanDoubleAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByBooleanDoubleAggregator = {
    val rva = new RegionValueTakeByBooleanDoubleAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByBooleanAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else SafeRow.read(keyType, region, k))
  }

  override def newInstance(): RegionValueTakeByBooleanAnnotationAggregator = new RegionValueTakeByBooleanAnnotationAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByBooleanAnnotationAggregator = {
    val rva = new RegionValueTakeByBooleanAnnotationAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByIntBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByIntBooleanAggregator = new RegionValueTakeByIntBooleanAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByIntBooleanAggregator = {
    val rva = new RegionValueTakeByIntBooleanAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByIntIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByIntIntAggregator = new RegionValueTakeByIntIntAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByIntIntAggregator = {
    val rva = new RegionValueTakeByIntIntAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByIntLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByIntLongAggregator = new RegionValueTakeByIntLongAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByIntLongAggregator = {
    val rva = new RegionValueTakeByIntLongAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByIntFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByIntFloatAggregator = new RegionValueTakeByIntFloatAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByIntFloatAggregator = {
    val rva = new RegionValueTakeByIntFloatAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByIntDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByIntDoubleAggregator = new RegionValueTakeByIntDoubleAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByIntDoubleAggregator = {
    val rva = new RegionValueTakeByIntDoubleAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByIntAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else SafeRow.read(keyType, region, k))
  }

  override def newInstance(): RegionValueTakeByIntAnnotationAggregator = new RegionValueTakeByIntAnnotationAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByIntAnnotationAggregator = {
    val rva = new RegionValueTakeByIntAnnotationAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByLongBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByLongBooleanAggregator = new RegionValueTakeByLongBooleanAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByLongBooleanAggregator = {
    val rva = new RegionValueTakeByLongBooleanAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByLongIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByLongIntAggregator = new RegionValueTakeByLongIntAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByLongIntAggregator = {
    val rva = new RegionValueTakeByLongIntAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByLongLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByLongLongAggregator = new RegionValueTakeByLongLongAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByLongLongAggregator = {
    val rva = new RegionValueTakeByLongLongAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByLongFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByLongFloatAggregator = new RegionValueTakeByLongFloatAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByLongFloatAggregator = {
    val rva = new RegionValueTakeByLongFloatAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByLongDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByLongDoubleAggregator = new RegionValueTakeByLongDoubleAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByLongDoubleAggregator = {
    val rva = new RegionValueTakeByLongDoubleAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByLongAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else SafeRow.read(keyType, region, k))
  }

  override def newInstance(): RegionValueTakeByLongAnnotationAggregator = new RegionValueTakeByLongAnnotationAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByLongAnnotationAggregator = {
    val rva = new RegionValueTakeByLongAnnotationAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByFloatBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByFloatBooleanAggregator = new RegionValueTakeByFloatBooleanAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByFloatBooleanAggregator = {
    val rva = new RegionValueTakeByFloatBooleanAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByFloatIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByFloatIntAggregator = new RegionValueTakeByFloatIntAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByFloatIntAggregator = {
    val rva = new RegionValueTakeByFloatIntAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByFloatLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByFloatLongAggregator = new RegionValueTakeByFloatLongAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByFloatLongAggregator = {
    val rva = new RegionValueTakeByFloatLongAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByFloatFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByFloatFloatAggregator = new RegionValueTakeByFloatFloatAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByFloatFloatAggregator = {
    val rva = new RegionValueTakeByFloatFloatAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByFloatDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByFloatDoubleAggregator = new RegionValueTakeByFloatDoubleAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByFloatDoubleAggregator = {
    val rva = new RegionValueTakeByFloatDoubleAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByFloatAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else SafeRow.read(keyType, region, k))
  }

  override def newInstance(): RegionValueTakeByFloatAnnotationAggregator = new RegionValueTakeByFloatAnnotationAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByFloatAnnotationAggregator = {
    val rva = new RegionValueTakeByFloatAnnotationAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByDoubleBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByDoubleBooleanAggregator = new RegionValueTakeByDoubleBooleanAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByDoubleBooleanAggregator = {
    val rva = new RegionValueTakeByDoubleBooleanAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByDoubleIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByDoubleIntAggregator = new RegionValueTakeByDoubleIntAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByDoubleIntAggregator = {
    val rva = new RegionValueTakeByDoubleIntAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByDoubleLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByDoubleLongAggregator = new RegionValueTakeByDoubleLongAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByDoubleLongAggregator = {
    val rva = new RegionValueTakeByDoubleLongAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByDoubleFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByDoubleFloatAggregator = new RegionValueTakeByDoubleFloatAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByDoubleFloatAggregator = {
    val rva = new RegionValueTakeByDoubleFloatAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByDoubleDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByDoubleDoubleAggregator = new RegionValueTakeByDoubleDoubleAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByDoubleDoubleAggregator = {
    val rva = new RegionValueTakeByDoubleDoubleAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByDoubleAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else SafeRow.read(keyType, region, k))
  }

  override def newInstance(): RegionValueTakeByDoubleAnnotationAggregator = new RegionValueTakeByDoubleAnnotationAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByDoubleAnnotationAggregator = {
    val rva = new RegionValueTakeByDoubleAnnotationAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByAnnotationBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByAnnotationBooleanAggregator = new RegionValueTakeByAnnotationBooleanAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByAnnotationBooleanAggregator = {
    val rva = new RegionValueTakeByAnnotationBooleanAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByAnnotationIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByAnnotationIntAggregator = new RegionValueTakeByAnnotationIntAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByAnnotationIntAggregator = {
    val rva = new RegionValueTakeByAnnotationIntAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByAnnotationLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByAnnotationLongAggregator = new RegionValueTakeByAnnotationLongAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByAnnotationLongAggregator = {
    val rva = new RegionValueTakeByAnnotationLongAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByAnnotationFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByAnnotationFloatAggregator = new RegionValueTakeByAnnotationFloatAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByAnnotationFloatAggregator = {
    val rva = new RegionValueTakeByAnnotationFloatAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByAnnotationDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else k)
  }

  override def newInstance(): RegionValueTakeByAnnotationDoubleAggregator = new RegionValueTakeByAnnotationDoubleAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByAnnotationDoubleAggregator = {
    val rva = new RegionValueTakeByAnnotationDoubleAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByAnnotationAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else SafeRow.read(keyType, region, k))
  }

  override def newInstance(): RegionValueTakeByAnnotationAnnotationAggregator = new RegionValueTakeByAnnotationAnnotationAggregator(n, aggType, keyType)

  override def copy(): RegionValueTakeByAnnotationAnnotationAggregator = {
    val rva = new RegionValueTakeByAnnotationAnnotationAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}