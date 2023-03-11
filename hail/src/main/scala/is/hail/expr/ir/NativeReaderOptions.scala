package is.hail.expr.ir

import is.hail.backend.HailStateManager
import is.hail.types.virtual._
import is.hail.expr.JSONAnnotationImpex
import is.hail.rvd.RVDPartitioner
import is.hail.utils._
import org.json4s.{CustomSerializer, DefaultFormats, Formats, JObject, JValue}
import org.json4s.JsonDSL._

class NativeReaderOptionsSerializer() extends CustomSerializer[NativeReaderOptions](
  format =>
    ({ case jObj: JObject =>
      implicit val fmt = format
      val filterIntervals = (jObj \ "filterIntervals").extract[Boolean]
      val intervalPointType = IRParser.parseStructType((jObj \ "intervalPointType").extract[String])
      val intervals = {
        val jv = jObj \ "intervals"
        val ty = TArray(TInterval(intervalPointType))
        JSONAnnotationImpex.importAnnotation(jv, ty).asInstanceOf[IndexedSeq[Interval]]
      }
      NativeReaderOptions(intervals, intervalPointType, filterIntervals)
    }, { case opts: NativeReaderOptions =>
      implicit val fmt = format
      val ty = TArray(TInterval(opts.intervalPointType))
      (("name" -> opts.getClass.getSimpleName) ~
        ("intervals" -> JSONAnnotationImpex.exportAnnotation(opts.intervals, ty)) ~
        ("intervalPointType" -> opts.intervalPointType.parsableString()) ~
        ("filterIntervals" -> opts.filterIntervals))
    })
)

object NativeReaderOptions {
  def fromJValue(jv: JValue): NativeReaderOptions = {
    implicit val formats: Formats = DefaultFormats

    val filterIntervals = (jv \ "filterIntervals").extract[Boolean]
    val intervalPointType = IRParser.parseStructType((jv \ "intervalPointType").extract[String])
    val intervals = {
      val jvIntervals = jv \ "intervals"
      val ty = TArray(TInterval(intervalPointType))
      JSONAnnotationImpex.importAnnotation(jvIntervals, ty).asInstanceOf[IndexedSeq[Interval]]
    }
    NativeReaderOptions(intervals, intervalPointType, filterIntervals)
  }
}

case class NativeReaderOptions(
  intervals: IndexedSeq[Interval],
  intervalPointType: TStruct,
  filterIntervals: Boolean = false) {
  def toJson: JValue = {
    val ty = TArray(TInterval(intervalPointType))
    JObject(
      "name" -> "NativeReaderOptions",
      "intervals" -> JSONAnnotationImpex.exportAnnotation(intervals, ty),
      "intervalPointType" -> intervalPointType.parsableString(),
      "filterIntervals" -> filterIntervals)
  }

  def renderShort(): String = s"(IntervalRead: ${intervals.length} intervals, filter=${filterIntervals})"

  def partitioner(sm: HailStateManager, base: RVDPartitioner): RVDPartitioner = {
    val partFromIntervals = RVDPartitioner.generate(sm, intervalPointType, intervals).extendKey(base.kType)
    if (filterIntervals)
      base.copy(rangeBounds = base.rangeBounds.filter(partFromIntervals.queryInterval(_).nonEmpty))
    else
      partFromIntervals
  }
}
