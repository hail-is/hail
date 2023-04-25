// Do not move this to a different package without updating packages= in log4j2.properties
package is.hail

import java.io._
import java.util.concurrent.TimeUnit
import org.apache.logging.log4j.core._
import org.apache.logging.log4j.core.layout._
import org.apache.logging.log4j.core.appender._
import org.apache.logging.log4j.core.config._
import org.apache.logging.log4j.core.config.plugins._

object QoBOutputStreamManager {
  private[this] var _instance: QoBOutputStreamManager = null

  def instance: QoBOutputStreamManager = synchronized { _instance }

  def getInstance(layout: Layout[_]): QoBOutputStreamManager = synchronized {
    // we only support one QoBOutputStreamManager
    assert(_instance == null)
    _instance = new QoBOutputStreamManager(layout)
    return _instance
  }
}

class QoBOutputStreamManager(layout: Layout[_]) extends OutputStreamManager(
  null,
  "qob",
  layout,
  true
) {
  private[this] var filename: String = "/tmp/hail-qob-job.log"

  override def createOutputStream(): OutputStream = {
    if (filename == null) {
      null
    } else {
      System.err.println("Creating outputstream " + filename)
      new BufferedOutputStream(new FileOutputStream(filename))
    }
  }

  def changeFile(newFilename: String): Unit = {
    System.err.println("changing file " + newFilename)
    flush()
    if (hasOutputStream()) {
      val os = getOutputStream()
      os.flush()
      os.close()
    }
    filename = newFilename
    setOutputStream(createOutputStream())
  }
}

object QoBAppender {
  @PluginFactory
  def createAppender(
    @PluginAttribute("name") name: String,
    @PluginAttribute("ignoreExceptions") ignoreExceptions: Boolean,
    @PluginElement("Layout") layout: Layout[_],
    @PluginElement("Filters") filter: Filter
  ): QoBAppender = {
    return new QoBAppender(name, ignoreExceptions, layout, filter)
  }
}

@Plugin(name = "QoBAppender", category = "Core", elementType = "appender", printObject = true)
class QoBAppender(
  name: String,
  ignoreExceptions: Boolean,
  layout: Layout[_],
  filter: Filter
) extends AbstractOutputStreamAppender[QoBOutputStreamManager](
  name,
  layout,
  filter,
  ignoreExceptions,
  false,
  Array[Property](),
  QoBOutputStreamManager.getInstance(layout)
) {
  override def append(event: LogEvent): Unit = {
    super.append(event)
    System.err.println("append " + layout.toSerializable(event))
  }
}

// object HailQoBJobAppender {
//   @PluginFactory
//   def createAppender(
//     @PluginAttribute("name") name: String,
//     @PluginAttribute("ignoreExceptions") ignoreExceptions: Boolean,
//     @PluginElement("Layout") layout: Layout[_],
//     @PluginElement("Filters") filter: Filter
//   ): HailQoBJobAppender = {
//     return new HailQoBJobAppender()
//   }

//   private var theOneHailQoBJobAppender: HailQoBJobAppender = null

//   def getTheOneHailQoBJobAppender: HailQoBJobAppender = theOneHailQoBJobAppender
// }

// @Plugin(name = "HailQoBJobAppender", category = "Core", elementType = "appender", printObject = true)
// class HailQoBJobAppender extends AbstractAppender(
//   "HailQoBJobAppender", null, PatternLayout.createDefaultLayout(), false, null
// ) {
//   private[this] val logFormat: String = "%d{yyyy-MM-dd HH:mm:ss.SSS} %c{1}: %p: %m%n"
//   private[this] val layout = PatternLayout.createDefaultLayout()
//   private[this] var fio = new PrintWriter(new BufferedOutputStream(new FileOutputStream("file1")))
//   System.err.println("I am alive")

//   if (HailQoBJobAppender.theOneHailQoBJobAppender != null) {
//     throw new RuntimeException("absolutely not.")
//   }

//   HailQoBJobAppender.theOneHailQoBJobAppender = this

//   override def append(event: LogEvent): Unit = {
//     val s = layout.toSerializable(event)
//     System.err.println("STDERR: " + s)
//     fio.write(s)
//   }

//   def flush() = fio.flush()

//   def close() = {
//     flush()
//     fio.close()
//   }

//   override def stop(timeout: Long, timeUnit: TimeUnit): Boolean = {
//     setStopping()
//     close()
//     setStopped()
//     return true
//   }

//   def requiresLayout(): Boolean = false

//   def changeFile(file: String): Unit = {
//     System.err.println("changeFile")
//     fio.flush()
//     fio.close()
//     fio = new PrintWriter(new BufferedOutputStream(new FileOutputStream(file)))
//   }
// }
