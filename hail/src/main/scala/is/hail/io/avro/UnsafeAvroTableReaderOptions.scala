package is.hail.io.avro

import is.hail.expr.JSONAnnotationImpex
import is.hail.expr.ir.IRParser
import is.hail.types.virtual.{TArray, TInterval, Type}
import is.hail.utils.Interval

import org.json4s.{CustomSerializer, Formats, JObject}
import org.json4s.JsonDSL._

case class UnsafeAvroTableReaderOptions(
  key: IndexedSeq[String],
  intervals: IndexedSeq[Interval],
  intervalPointType: Type,
)

class UnsafeAvroTableReaderOptionsSerializer
    extends CustomSerializer[UnsafeAvroTableReaderOptions](format =>
      (
        { case jv: JObject =>
          implicit val fmt: Formats = format
          val key = (jv \ "key").extract[IndexedSeq[String]]
          val intervalPointType = IRParser.parseType((jv \ "intervalPointType").extract[String])
          val intervals = {
            val jIntervals = jv \ "intervals"
            val ty = TArray(TInterval(intervalPointType))
            JSONAnnotationImpex.importAnnotation(jIntervals, ty).asInstanceOf[IndexedSeq[Interval]]
          }
          UnsafeAvroTableReaderOptions(key, intervals, intervalPointType)
        },
        { case UnsafeAvroTableReaderOptions(key, intervals, intervalPointType) =>
          val ty = TArray(TInterval(intervalPointType))
          ("name" -> UnsafeAvroTableReaderOptions.getClass.getSimpleName) ~
            ("key" -> key) ~
            ("intervals" -> JSONAnnotationImpex.exportAnnotation(intervals, ty)) ~
            ("intervalPointType" -> intervalPointType.parsableString())
        },
      )
    )
