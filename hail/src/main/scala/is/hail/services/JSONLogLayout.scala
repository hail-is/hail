package is.hail.services

import org.json4s._
import org.json4s.jackson.JsonMethods

import java.io.StringWriter
import java.text._
import java.util.function._
import scala.collection.mutable.ArrayBuffer

import org.apache.log4j._
import org.apache.log4j.spi._

class DateFormatter {
  private[this] val fmt = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS")
  private[this] val buffer = new StringBuffer()
  private[this] val fp = new FieldPosition(0)

  def format(milliseconds: Long): String = {
    buffer.setLength(0)
    fmt.format(milliseconds, buffer, fp)
    buffer.toString
  }
}

object JSONLogLayout {
  private val datefmt = ThreadLocal.withInitial(new Supplier[DateFormatter]() {
    def get() = new DateFormatter()
  })
}

class JSONLogLayout extends Layout {
  import JSONLogLayout._

  def ignoresThrowable(): Boolean = false

  def activateOptions(): Unit = ()

  def format(event: LoggingEvent): String = {
    val threadName = event.getThreadName();
    val timestamp = event.getTimeStamp();
    val mdc = event.getProperties();
    val ndc = event.getNDC();
    val throwableInfo = event.getThrowableInformation()
    val locationInfo = event.getLocationInformation()
    val fields = new ArrayBuffer[JField]()
    fields += JField("@version", JInt(1))
    fields += JField("@timestamp", JString(datefmt.get.format(timestamp)))
    fields += JField("message", JString(event.getRenderedMessage()))
    fields += JField("filename", JString(locationInfo.getFileName()))
    fields += JField("line_number", JString(locationInfo.getLineNumber()))
    fields += JField("class", JString(locationInfo.getClassName()))
    fields += JField("method", JString(locationInfo.getMethodName()))
    fields += JField("logger_name", JString(event.getLoggerName()))

    val mdcFields = new ArrayBuffer[JField]()
    mdc.forEach(new BiConsumer[Any, Any]() {
      def accept(key: Any, value: Any): Unit =
        mdcFields += JField(key.toString, JString(value.toString))
    })
    fields += JField("mdc", JObject(mdcFields: _*))

    fields += JField("ndc", JString(ndc))
    fields += JField("severity", JString(event.getLevel().toString()))
    fields += JField("thread_name", JString(threadName))

    if (throwableInfo != null) {
      fields += JField(
        "exception_class",
        JString(throwableInfo.getThrowable().getClass().getCanonicalName()),
      )
      fields += JField("exception_message", JString(throwableInfo.getThrowable().getMessage()))
      fields += JField(
        "exception_stacktrace",
        JString(formatException(throwableInfo.getThrowable())),
      )
    }
    val jsonEvent = JObject(fields: _*)

    val sw = new StringWriter()
    JsonMethods.mapper.writeValue(sw, jsonEvent)
    sw.append('\n')
    sw.toString()
  }
}
