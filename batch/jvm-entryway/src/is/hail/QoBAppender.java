// Do not move this to a different package without updating packages= in log4j2.properties
package is.hail;

import java.io.Serializable;
import org.apache.logging.log4j.core.Filter;
import org.apache.logging.log4j.core.Layout;
import org.apache.logging.log4j.core.appender.AbstractOutputStreamAppender;
import org.apache.logging.log4j.core.config.Property;
import org.apache.logging.log4j.core.config.plugins.Plugin;
import org.apache.logging.log4j.core.config.plugins.PluginAttribute;
import org.apache.logging.log4j.core.config.plugins.PluginElement;
import org.apache.logging.log4j.core.config.plugins.PluginFactory;

@Plugin(name = "QoBAppender", category = "Core", elementType = "appender", printObject = true)
public final class QoBAppender extends AbstractOutputStreamAppender<QoBOutputStreamManager> {
  @PluginFactory
  public static QoBAppender createAppender(
      @PluginAttribute("name") String name,
      @PluginAttribute("ignoreExceptions") boolean ignoreExceptions,
      @PluginAttribute("immediateFlush") boolean immediateFlush,
      @PluginElement("Layout") Layout<? extends Serializable> layout,
      @PluginElement("Filters") Filter filter) {
    return new QoBAppender(name, ignoreExceptions, immediateFlush, layout, filter);
  }

  private QoBAppender(
      String name,
      boolean ignoreExceptions,
      boolean immediateFlush,
      Layout<? extends Serializable> layout,
      Filter filter) {
    super(
        name,
        layout,
        filter,
        ignoreExceptions,
        immediateFlush,
        Property.EMPTY_ARRAY,
        QoBOutputStreamManager.getInstance(layout));
  }
}
