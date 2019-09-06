package is.hail.annotations.aggregators

import is.hail.annotations.{Annotation, Region, RegionValueBuilder, SafeRow}
import is.hail.expr.types.physical.PType
import is.hail.expr.types.virtual.Type
import is.hail.io._

class RegionValuePrevNonnullAnnotationAggregator2(
  t: PType,
  makeEncoder: (MemoryBuffer) => Encoder,
  makeDecoder: (MemoryBuffer) => Decoder
) extends RegionValueAggregator {
  def this(t: PType) = this(t, {
    val f = EmitPackEncoder(t, t)
    (mb: MemoryBuffer) => new CompiledEncoder(new MemoryOutputBuffer(mb), f)
  }, {
    val f = EmitPackDecoder(t, t)
    (mb: MemoryBuffer) => new CompiledDecoder(new MemoryInputBuffer(mb), f)
  })
  def this(t: Type) = this(t.physicalType)

  val mb = new MemoryBuffer
  @transient lazy val encoder: Encoder = makeEncoder(mb)
  @transient lazy val decoder: Decoder = makeDecoder(mb)

  var present: Boolean = false

  override def isCommutative: Boolean = false

  def seqOp(region: Region, offset: Long, missing: Boolean) {
    if (!missing) {
      mb.clear()
      encoder.writeRegionValue(region, offset)
      present = true
    }
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValuePrevNonnullAnnotationAggregator2]
    if (that.present) {
      mb.copyFrom(that.mb)
      present = true
    }
  }

  def result(rvb: RegionValueBuilder) {
    if (present) {
      mb.clearPos()
      val p = decoder.readRegionValue(rvb.region)
      rvb.addRegionValue(t, rvb.region, p)
    } else
      rvb.setMissing()
  }

  def newInstance(): RegionValuePrevNonnullAnnotationAggregator2 =
    new RegionValuePrevNonnullAnnotationAggregator2(t, makeEncoder, makeDecoder)

  def copy(): RegionValuePrevNonnullAnnotationAggregator2 = {
    val rva = new RegionValuePrevNonnullAnnotationAggregator2(t, makeEncoder, makeDecoder)
    if (present) {
      rva.mb.copyFrom(mb)
      rva.present = true
    }
    rva
  }

  def clear() {
    mb.clear()
    present = false
  }
}

class RegionValuePrevNonnullAnnotationAggregator(t: Type) extends RegionValueAggregator {
  var last: Annotation = null

  override def isCommutative: Boolean = false

  def seqOp(region: Region, offset: Long, missing: Boolean) {
    if (!missing)
      last = SafeRow.read(t.physicalType, region, offset)
  }

  def combOp(agg2: RegionValueAggregator) {
    val that = agg2.asInstanceOf[RegionValuePrevNonnullAnnotationAggregator]
    if (that.last != null)
      last = that.last
  }

  def result(rvb: RegionValueBuilder) {
    if (last != null)
      rvb.addAnnotation(t, last)
    else
      rvb.setMissing()
  }

  def newInstance(): RegionValuePrevNonnullAnnotationAggregator = new RegionValuePrevNonnullAnnotationAggregator(t)

  def copy(): RegionValuePrevNonnullAnnotationAggregator = {
    val rva = new RegionValuePrevNonnullAnnotationAggregator(t)
    rva.last = last
    rva
  }

  def clear() {
    last = null
  }
}
