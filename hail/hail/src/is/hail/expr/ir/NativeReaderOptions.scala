package is.hail.expr.ir

import is.hail.expr.JSONAnnotationImpex
import is.hail.types.TypeStringSerializer
import is.hail.types.virtual._
import is.hail.utils._
import is.hail.variant.ReferenceGenomeEnv

import org.json4s.{CustomSerializer, DefaultFormats, Formats, JObject, JValue}
import org.json4s.JsonDSL._

class NativeReaderOptionsSerializer(rgs: Option[ReferenceGenomeEnv])
    extends CustomSerializer[NativeReaderOptions](format =>
      (
        { case jObj: JObject =>
          implicit val fmt = format
          val filterIntervals = (jObj \ "filterIntervals").extract[Boolean]
          val intervalPointType = IRParser.parseType(
            (jObj \ "intervalPointType").extract[String],
            rgs.getOrElse(TypeStringSerializer.cannotRead[NativeReaderOptions]),
          )
          val intervals = {
            val jv = jObj \ "intervals"
            val ty = TArray(TInterval(intervalPointType))
            JSONAnnotationImpex.importAnnotation(jv, ty).asInstanceOf[IndexedSeq[Interval]]
          }
          NativeReaderOptions(intervals, intervalPointType, filterIntervals)
        },
        { case opts: NativeReaderOptions =>
          val ty = TArray(TInterval(opts.intervalPointType))
          ("name" -> opts.getClass.getSimpleName) ~
            ("intervals" -> JSONAnnotationImpex.exportAnnotation(opts.intervals, ty)) ~
            ("intervalPointType" -> opts.intervalPointType.parsableString()) ~
            ("filterIntervals" -> opts.filterIntervals)
        },
      )
    ) {
  def this(rgs: ReferenceGenomeEnv) = this(Some(rgs))
}

object NativeReaderOptions {
  def fromJValue(jv: JValue, rgs: ReferenceGenomeEnv): NativeReaderOptions = {
    implicit val formats: Formats = DefaultFormats

    val filterIntervals = (jv \ "filterIntervals").extract[Boolean]
    val intervalPointType = IRParser.parseType((jv \ "intervalPointType").extract[String], rgs)
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
