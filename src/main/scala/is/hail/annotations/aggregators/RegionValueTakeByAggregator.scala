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

  override def copy(): RegionValueTakeByBooleanBooleanAggregator = new RegionValueTakeByBooleanBooleanAggregator(n, aggType, keyType)
}

class RegionValueTakeByBooleanIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByBooleanIntAggregator = new RegionValueTakeByBooleanIntAggregator(n, aggType, keyType)
}

class RegionValueTakeByBooleanLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByBooleanLongAggregator = new RegionValueTakeByBooleanLongAggregator(n, aggType, keyType)
}

class RegionValueTakeByBooleanFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByBooleanFloatAggregator = new RegionValueTakeByBooleanFloatAggregator(n, aggType, keyType)
}

class RegionValueTakeByBooleanDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByBooleanDoubleAggregator = new RegionValueTakeByBooleanDoubleAggregator(n, aggType, keyType)
}

class RegionValueTakeByBooleanAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Boolean, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else SafeRow.read(keyType, region, k))
  }

  override def copy(): RegionValueTakeByBooleanAnnotationAggregator = new RegionValueTakeByBooleanAnnotationAggregator(n, aggType, keyType)
}

class RegionValueTakeByIntBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByIntBooleanAggregator = new RegionValueTakeByIntBooleanAggregator(n, aggType, keyType)
}

class RegionValueTakeByIntIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByIntIntAggregator = new RegionValueTakeByIntIntAggregator(n, aggType, keyType)
}

class RegionValueTakeByIntLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByIntLongAggregator = new RegionValueTakeByIntLongAggregator(n, aggType, keyType)
}

class RegionValueTakeByIntFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByIntFloatAggregator = new RegionValueTakeByIntFloatAggregator(n, aggType, keyType)
}

class RegionValueTakeByIntDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByIntDoubleAggregator = new RegionValueTakeByIntDoubleAggregator(n, aggType, keyType)
}

class RegionValueTakeByIntAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Int, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else SafeRow.read(keyType, region, k))
  }

  override def copy(): RegionValueTakeByIntAnnotationAggregator = new RegionValueTakeByIntAnnotationAggregator(n, aggType, keyType)
}

class RegionValueTakeByLongBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByLongBooleanAggregator = new RegionValueTakeByLongBooleanAggregator(n, aggType, keyType)
}

class RegionValueTakeByLongIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByLongIntAggregator = new RegionValueTakeByLongIntAggregator(n, aggType, keyType)
}

class RegionValueTakeByLongLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByLongLongAggregator = new RegionValueTakeByLongLongAggregator(n, aggType, keyType)
}

class RegionValueTakeByLongFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByLongFloatAggregator = new RegionValueTakeByLongFloatAggregator(n, aggType, keyType)
}

class RegionValueTakeByLongDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByLongDoubleAggregator = new RegionValueTakeByLongDoubleAggregator(n, aggType, keyType)
}

class RegionValueTakeByLongAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else SafeRow.read(keyType, region, k))
  }

  override def copy(): RegionValueTakeByLongAnnotationAggregator = new RegionValueTakeByLongAnnotationAggregator(n, aggType, keyType)
}

class RegionValueTakeByFloatBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByFloatBooleanAggregator = new RegionValueTakeByFloatBooleanAggregator(n, aggType, keyType)
}

class RegionValueTakeByFloatIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByFloatIntAggregator = new RegionValueTakeByFloatIntAggregator(n, aggType, keyType)
}

class RegionValueTakeByFloatLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByFloatLongAggregator = new RegionValueTakeByFloatLongAggregator(n, aggType, keyType)
}

class RegionValueTakeByFloatFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByFloatFloatAggregator = new RegionValueTakeByFloatFloatAggregator(n, aggType, keyType)
}

class RegionValueTakeByFloatDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByFloatDoubleAggregator = new RegionValueTakeByFloatDoubleAggregator(n, aggType, keyType)
}

class RegionValueTakeByFloatAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Float, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else SafeRow.read(keyType, region, k))
  }

  override def copy(): RegionValueTakeByFloatAnnotationAggregator = new RegionValueTakeByFloatAnnotationAggregator(n, aggType, keyType)
}

class RegionValueTakeByDoubleBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByDoubleBooleanAggregator = new RegionValueTakeByDoubleBooleanAggregator(n, aggType, keyType)
}

class RegionValueTakeByDoubleIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByDoubleIntAggregator = new RegionValueTakeByDoubleIntAggregator(n, aggType, keyType)
}

class RegionValueTakeByDoubleLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByDoubleLongAggregator = new RegionValueTakeByDoubleLongAggregator(n, aggType, keyType)
}

class RegionValueTakeByDoubleFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByDoubleFloatAggregator = new RegionValueTakeByDoubleFloatAggregator(n, aggType, keyType)
}

class RegionValueTakeByDoubleDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else k)
  }

  override def copy(): RegionValueTakeByDoubleDoubleAggregator = new RegionValueTakeByDoubleDoubleAggregator(n, aggType, keyType)
}

class RegionValueTakeByDoubleAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Double, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else x, if (km) null else SafeRow.read(keyType, region, k))
  }

  override def copy(): RegionValueTakeByDoubleAnnotationAggregator = new RegionValueTakeByDoubleAnnotationAggregator(n, aggType, keyType)
}

class RegionValueTakeByAnnotationBooleanAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Boolean, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else k)
  }

  override def copy(): RegionValueTakeByAnnotationBooleanAggregator = new RegionValueTakeByAnnotationBooleanAggregator(n, aggType, keyType)
}

class RegionValueTakeByAnnotationIntAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Int, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else k)
  }

  override def copy(): RegionValueTakeByAnnotationIntAggregator = new RegionValueTakeByAnnotationIntAggregator(n, aggType, keyType)
}

class RegionValueTakeByAnnotationLongAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else k)
  }

  override def copy(): RegionValueTakeByAnnotationLongAggregator = new RegionValueTakeByAnnotationLongAggregator(n, aggType, keyType)
}

class RegionValueTakeByAnnotationFloatAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Float, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else k)
  }

  override def copy(): RegionValueTakeByAnnotationFloatAggregator = new RegionValueTakeByAnnotationFloatAggregator(n, aggType, keyType)
}

class RegionValueTakeByAnnotationDoubleAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Double, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else k)
  }

  override def copy(): RegionValueTakeByAnnotationDoubleAggregator = new RegionValueTakeByAnnotationDoubleAggregator(n, aggType, keyType)
}

class RegionValueTakeByAnnotationAnnotationAggregator(n: Int, aggType: Type, keyType: Type) extends RegionValueTakeByAggregator(n, aggType, keyType) {
  def seqOp(region: Region, x: Long, xm: Boolean, k: Long, km: Boolean) {
    seqOp(if (xm) null else SafeRow.read(aggType, region, x), if (km) null else SafeRow.read(keyType, region, k))
  }

  override def copy(): RegionValueTakeByAnnotationAnnotationAggregator = new RegionValueTakeByAnnotationAnnotationAggregator(n, aggType, keyType)
}