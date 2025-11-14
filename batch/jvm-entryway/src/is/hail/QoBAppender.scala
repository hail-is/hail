// Do not move this to a different package without updating packages= in log4j2.properties
package is.hail

import scala.collection.mutable

import java.io.{BufferedOutputStream, FileOutputStream, OutputStream}

import org.apache.logging.log4j.core.{Filter, Layout}
import org.apache.logging.log4j.core.appender.{AbstractOutputStreamAppender, OutputStreamManager}
import org.apache.logging.log4j.core.config.Property
import org.apache.logging.log4j.core.config.plugins.{
  Plugin, PluginAttribute, PluginElement, PluginFactory,
}

object QoBOutputStreamManager {
  private[this] var _instances: mutable.Map[Layout[_], QoBOutputStreamManager] = mutable.Map()
  private[this] var _filename: String = _

  def getInstance(layout: Layout[_]): QoBOutputStreamManager = synchronized {
    _instances.getOrElseUpdate(layout, new QoBOutputStreamManager(layout, _filename))
  }

  def changeFileInAllAppenders(newFilename: String): Unit = synchronized {
    _filename = newFilename
    _instances.values.foreach(_.changeFile(newFilename))
  }

  private def remove(layout: Layout[_]): Unit = synchronized {
    val _ = _instances.remove(layout)
  }

  def flushAllAppenders(): Unit = synchronized {
    _instances.values.foreach(_.flush())
  }
}

class QoBOutputStreamManager(
  layout: Layout[_],
  private[this] var filename: String,
) extends OutputStreamManager(
      null,
      "QoBOutputStreamManager",
      layout,
      true,
    ) {
  override def createOutputStream(): OutputStream = {
    assert(filename != null)
    new BufferedOutputStream(new FileOutputStream(filename))
  }

  override def close(): Unit = {
    super.close()
    QoBOutputStreamManager.remove(layout)
  }

  private def changeFile(newFilename: String): Unit = {
    if (hasOutputStream) {
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
    @PluginElement("Layout") layout: Layout[_ <: Serializable],
    @PluginElement("Filters") filter: Filter,
  ): QoBAppender =
    new QoBAppender(name, ignoreExceptions, immediateFlush, layout, filter)
}

@Plugin(
  name = "QoBAppender",
  category = "Core",
  elementType = "appender",
  printObject = true,
)
class QoBAppender(
  name: String,
  ignoreExceptions: Boolean,
  immediateFlush: Boolean,
  layout: Layout[_ <: Serializable],
  filter: Filter,
) extends AbstractOutputStreamAppender[QoBOutputStreamManager](
      name,
      layout,
      filter,
      ignoreExceptions,
      immediateFlush,
      Array[Property](),
      QoBOutputStreamManager.getInstance(layout),
    )
