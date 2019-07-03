package is.hail.expr.ir

import is.hail.annotations._
import is.hail.expr.types.virtual._
import is.hail.expr.JSONAnnotationImpex
import is.hail.utils._
import org.json4s.{Formats, ShortTypeHints, CustomSerializer, JObject}
import org.json4s.JsonAST.{JArray, JInt, JNull, JString, JField, JNothing}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods

class NativeReaderOptionsSerializer() extends CustomSerializer[NativeReaderOptions](
  format =>
    ({ case jObj: JObject =>
      implicit val fmt = format
      val filterIntervals = (jObj \ "filterIntervals").extract[Boolean]
      val intervalPointType = IRParser.parseType((jObj \ "intervalPointType").extract[String])
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

case class NativeReaderOptions(
  intervals: IndexedSeq[Interval],
  intervalPointType: Type,
  filterIntervals: Boolean = false
)
