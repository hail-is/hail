// Do not move this to a different package without updating packages= in log4j2.properties
package is.hail

import java.io._
import java.util.concurrent.TimeUnit
import org.apache.logging.log4j.core._
import org.apache.logging.log4j.core.layout._
import org.apache.logging.log4j.core.appender._
import org.apache.logging.log4j.core.config._
import org.apache.logging.log4j.core.config.plugins._
import scala.collection.mutable

object QoBOutputStreamManager {
  private[this] var _instances: mutable.Map[Layout[_], QoBOutputStreamManager] = mutable.Map()

  def getInstance(layout: Layout[_]): QoBOutputStreamManager = synchronized {
    _instances.getOrElseUpdate(layout, new QoBOutputStreamManager(layout))
  }

  def changeFileInAllAppenders(newFilename: String): Unit = synchronized {
    _instances.values.foreach(_.changeFile(newFilename))
  }

  def remove(layout: Layout[_]): Unit = synchronized {
    _instances.remove(layout)
  }

  def flushAllAppenders(): Unit = synchronized {
    _instances.values.foreach(_.flush())
  }
}

class QoBOutputStreamManager(layout: Layout[_]) extends OutputStreamManager(
  null,
  "QoBOutputStreamManager",
  layout,
  true
) {
  private[this] var filename: String = null

  override def createOutputStream(): OutputStream = {
    assert(filename != null)
    new BufferedOutputStream(new FileOutputStream(filename))
  }

  override def close(): Unit = {
    super.close()
    QoBOutputStreamManager.remove(layout)
  }

  def changeFile(newFilename: String): Unit = {
    if (hasOutputStream()) {
      closeOutputStream()
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
    @PluginAttribute("immediateFlush") immediateFlush: Boolean,
    @PluginElement("Layout") layout: Layout[_],
    @PluginElement("Filters") filter: Filter
  ): QoBAppender = {
    return new QoBAppender(name, ignoreExceptions, immediateFlush, layout, filter)
  }
}

@Plugin(name = "QoBAppender", category = "Core", elementType = "appender", printObject = true)
class QoBAppender(
  name: String,
  ignoreExceptions: Boolean,
  immediateFlush: Boolean,
  layout: Layout[_],
  filter: Filter
) extends AbstractOutputStreamAppender[QoBOutputStreamManager](
  name,
  layout,
  filter,
  ignoreExceptions,
  immediateFlush,
  Array[Property](),
  QoBOutputStreamManager.getInstance(layout)
) {
}
