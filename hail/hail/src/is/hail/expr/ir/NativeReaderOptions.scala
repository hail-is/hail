package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.expr.JSONAnnotationImpex
import is.hail.types.virtual._
import is.hail.utils._
import org.json4s.{CustomSerializer, DefaultFormats, Formats, JObject, JValue, Serializer}
import org.json4s.JsonDSL._

object NativeReaderOptionsSerializer {
  def apply(ctx: ExecuteContext): Serializer[NativeReaderOptions] =
    new CustomSerializer[NativeReaderOptions](implicit format =>
      (
        {
          case jObj: JObject =>
            val filterIntervals = (jObj \ "filterIntervals").extract[Boolean]
            val intervalPointType =
              IRParser.parseType(ctx, (jObj \ "intervalPointType").extract[String])
            val intervals = {
              val jv = jObj \ "intervals"
              val ty = TArray(TInterval(intervalPointType))
              JSONAnnotationImpex.importAnnotation(jv, ty).asInstanceOf[IndexedSeq[Interval]]
            }
            NativeReaderOptions(intervals, intervalPointType, filterIntervals)
        },
        {
          case opts: NativeReaderOptions =>
            val ty = TArray(TInterval(opts.intervalPointType))
            ("name" -> opts.getClass.getSimpleName) ~
              ("intervals" -> JSONAnnotationImpex.exportAnnotation(opts.intervals, ty)) ~
              ("intervalPointType" -> opts.intervalPointType.parsableString()) ~
              ("filterIntervals" -> opts.filterIntervals)
        },
      )
    )
}

object NativeReaderOptions {
  def fromJValue(ctx: ExecuteContext, jv: JValue): NativeReaderOptions = {
    implicit val formats: Formats = DefaultFormats

    val filterIntervals = (jv \ "filterIntervals").extract[Boolean]
    val intervalPointType = IRParser.parseType(ctx, (jv \ "intervalPointType").extract[String])
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
  intervalPointType: Type,
  filterIntervals: Boolean = false,
) {
  def toJson: JValue = {
    val ty = TArray(TInterval(intervalPointType))
    JObject(
      "name" -> "NativeReaderOptions",
      "intervals" -> JSONAnnotationImpex.exportAnnotation(intervals, ty),
      "intervalPointType" -> intervalPointType.parsableString(),
      "filterIntervals" -> filterIntervals,
    )
  }

  def renderShort(): String =
    s"(IntervalRead: ${intervals.length} intervals, filter=$filterIntervals)"
}
