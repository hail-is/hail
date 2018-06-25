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

  override def copy(): RegionValueTakeByAggregator = new RegionValueTakeByAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByAggregator = {
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

  override def copy(): RegionValueTakeByBooleanBooleanAggregator = new RegionValueTakeByBooleanBooleanAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByBooleanBooleanAggregator = {
    val rva = new RegionValueTakeByBooleanBooleanAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByBooleanIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByBooleanIntAggregator = new RegionValueTakeByBooleanIntAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByBooleanIntAggregator = {
    val rva = new RegionValueTakeByBooleanIntAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByBooleanLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByBooleanLongAggregator = new RegionValueTakeByBooleanLongAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByBooleanLongAggregator = {
    val rva = new RegionValueTakeByBooleanLongAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByBooleanFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByBooleanFloatAggregator = new RegionValueTakeByBooleanFloatAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByBooleanFloatAggregator = {
    val rva = new RegionValueTakeByBooleanFloatAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByBooleanDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByBooleanDoubleAggregator = new RegionValueTakeByBooleanDoubleAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByBooleanDoubleAggregator = {
    val rva = new RegionValueTakeByBooleanDoubleAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByBooleanAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else SafeRow.read(keyType, region, k))
  }

  override def copy(): RegionValueTakeByBooleanAnnotationAggregator = new RegionValueTakeByBooleanAnnotationAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByBooleanAnnotationAggregator = {
    val rva = new RegionValueTakeByBooleanAnnotationAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByIntBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByIntBooleanAggregator = new RegionValueTakeByIntBooleanAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByIntBooleanAggregator = {
    val rva = new RegionValueTakeByIntBooleanAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByIntIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByIntIntAggregator = new RegionValueTakeByIntIntAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByIntIntAggregator = {
    val rva = new RegionValueTakeByIntIntAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByIntLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByIntLongAggregator = new RegionValueTakeByIntLongAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByIntLongAggregator = {
    val rva = new RegionValueTakeByIntLongAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByIntFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByIntFloatAggregator = new RegionValueTakeByIntFloatAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByIntFloatAggregator = {
    val rva = new RegionValueTakeByIntFloatAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByIntDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByIntDoubleAggregator = new RegionValueTakeByIntDoubleAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByIntDoubleAggregator = {
    val rva = new RegionValueTakeByIntDoubleAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByIntAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else SafeRow.read(keyType, region, k))
  }

  override def copy(): RegionValueTakeByIntAnnotationAggregator = new RegionValueTakeByIntAnnotationAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByIntAnnotationAggregator = {
    val rva = new RegionValueTakeByIntAnnotationAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByLongBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByLongBooleanAggregator = new RegionValueTakeByLongBooleanAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByLongBooleanAggregator = {
    val rva = new RegionValueTakeByLongBooleanAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByLongIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByLongIntAggregator = new RegionValueTakeByLongIntAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByLongIntAggregator = {
    val rva = new RegionValueTakeByLongIntAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByLongLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByLongLongAggregator = new RegionValueTakeByLongLongAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByLongLongAggregator = {
    val rva = new RegionValueTakeByLongLongAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByLongFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByLongFloatAggregator = new RegionValueTakeByLongFloatAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByLongFloatAggregator = {
    val rva = new RegionValueTakeByLongFloatAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByLongDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByLongDoubleAggregator = new RegionValueTakeByLongDoubleAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByLongDoubleAggregator = {
    val rva = new RegionValueTakeByLongDoubleAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByLongAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else SafeRow.read(keyType, region, k))
  }

  override def copy(): RegionValueTakeByLongAnnotationAggregator = new RegionValueTakeByLongAnnotationAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByLongAnnotationAggregator = {
    val rva = new RegionValueTakeByLongAnnotationAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByFloatBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByFloatBooleanAggregator = new RegionValueTakeByFloatBooleanAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByFloatBooleanAggregator = {
    val rva = new RegionValueTakeByFloatBooleanAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByFloatIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByFloatIntAggregator = new RegionValueTakeByFloatIntAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByFloatIntAggregator = {
    val rva = new RegionValueTakeByFloatIntAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByFloatLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByFloatLongAggregator = new RegionValueTakeByFloatLongAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByFloatLongAggregator = {
    val rva = new RegionValueTakeByFloatLongAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByFloatFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByFloatFloatAggregator = new RegionValueTakeByFloatFloatAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByFloatFloatAggregator = {
    val rva = new RegionValueTakeByFloatFloatAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByFloatDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByFloatDoubleAggregator = new RegionValueTakeByFloatDoubleAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByFloatDoubleAggregator = {
    val rva = new RegionValueTakeByFloatDoubleAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByFloatAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else SafeRow.read(keyType, region, k))
  }

  override def copy(): RegionValueTakeByFloatAnnotationAggregator = new RegionValueTakeByFloatAnnotationAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByFloatAnnotationAggregator = {
    val rva = new RegionValueTakeByFloatAnnotationAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByDoubleBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByDoubleBooleanAggregator = new RegionValueTakeByDoubleBooleanAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByDoubleBooleanAggregator = {
    val rva = new RegionValueTakeByDoubleBooleanAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByDoubleIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByDoubleIntAggregator = new RegionValueTakeByDoubleIntAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByDoubleIntAggregator = {
    val rva = new RegionValueTakeByDoubleIntAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByDoubleLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByDoubleLongAggregator = new RegionValueTakeByDoubleLongAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByDoubleLongAggregator = {
    val rva = new RegionValueTakeByDoubleLongAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByDoubleFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByDoubleFloatAggregator = new RegionValueTakeByDoubleFloatAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByDoubleFloatAggregator = {
    val rva = new RegionValueTakeByDoubleFloatAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByDoubleDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByDoubleDoubleAggregator = new RegionValueTakeByDoubleDoubleAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByDoubleDoubleAggregator = {
    val rva = new RegionValueTakeByDoubleDoubleAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByDoubleAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else SafeRow.read(keyType, region, k))
  }

  override def copy(): RegionValueTakeByDoubleAnnotationAggregator = new RegionValueTakeByDoubleAnnotationAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByDoubleAnnotationAggregator = {
    val rva = new RegionValueTakeByDoubleAnnotationAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByAnnotationBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else k)
  }

  override def copy(): RegionValueTakeByAnnotationBooleanAggregator = new RegionValueTakeByAnnotationBooleanAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByAnnotationBooleanAggregator = {
    val rva = new RegionValueTakeByAnnotationBooleanAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByAnnotationIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else k)
  }

  override def copy(): RegionValueTakeByAnnotationIntAggregator = new RegionValueTakeByAnnotationIntAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByAnnotationIntAggregator = {
    val rva = new RegionValueTakeByAnnotationIntAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByAnnotationLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else k)
  }

  override def copy(): RegionValueTakeByAnnotationLongAggregator = new RegionValueTakeByAnnotationLongAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByAnnotationLongAggregator = {
    val rva = new RegionValueTakeByAnnotationLongAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByAnnotationFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else k)
  }

  override def copy(): RegionValueTakeByAnnotationFloatAggregator = new RegionValueTakeByAnnotationFloatAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByAnnotationFloatAggregator = {
    val rva = new RegionValueTakeByAnnotationFloatAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByAnnotationDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else k)
  }

  override def copy(): RegionValueTakeByAnnotationDoubleAggregator = new RegionValueTakeByAnnotationDoubleAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByAnnotationDoubleAggregator = {
    val rva = new RegionValueTakeByAnnotationDoubleAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}

class RegionValueTakeByAnnotationAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else SafeRow.read(keyType, region, k))
  }

  override def copy(): RegionValueTakeByAnnotationAnnotationAggregator = new RegionValueTakeByAnnotationAnnotationAggregator(n, aggType, keyType)

  override def deepCopy(): RegionValueTakeByAnnotationAnnotationAggregator = {
    val rva = new RegionValueTakeByAnnotationAnnotationAggregator(n, aggType, keyType)
    rva._state = _state.clone()
    rva
  }
}