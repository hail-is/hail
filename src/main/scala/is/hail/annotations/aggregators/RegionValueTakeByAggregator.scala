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

  override def clear() {
    _state.clear()
  }
}

class RegionValueTakeByBooleanBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByBooleanIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByBooleanLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByBooleanFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByBooleanDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByBooleanAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else SafeRow.read(keyType, region, k))
  }
}

class RegionValueTakeByIntBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByIntIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByIntLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByIntFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByIntDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByIntAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else SafeRow.read(keyType, region, k))
  }
}

class RegionValueTakeByLongBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByLongIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByLongLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByLongFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByLongDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByLongAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else SafeRow.read(keyType, region, k))
  }
}

class RegionValueTakeByFloatBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByFloatIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByFloatLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByFloatFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByFloatDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByFloatAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else SafeRow.read(keyType, region, k))
  }
}

class RegionValueTakeByDoubleBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByDoubleIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByDoubleLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByDoubleFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByDoubleDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }
}

class RegionValueTakeByDoubleAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else SafeRow.read(keyType, region, k))
  }
}

class RegionValueTakeByAnnotationBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else k)
  }
}

class RegionValueTakeByAnnotationIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else k)
  }
}

class RegionValueTakeByAnnotationLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else k)
  }
}

class RegionValueTakeByAnnotationFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else k)
  }
}

class RegionValueTakeByAnnotationDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else k)
  }
}

class RegionValueTakeByAnnotationAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else SafeRow.read(keyType, region, k))
  }
}