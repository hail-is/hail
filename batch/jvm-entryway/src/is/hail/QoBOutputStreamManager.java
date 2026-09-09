package is.hail;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import org.apache.logging.log4j.core.Layout;
import org.apache.logging.log4j.core.appender.OutputStreamManager;

final class QoBOutputStreamManager extends OutputStreamManager {
  private static final Map<Layout<?>, QoBOutputStreamManager> instances = new HashMap<>();
  private static String filename;

  static synchronized QoBOutputStreamManager getInstance(Layout<?> layout) {
    return instances.computeIfAbsent(layout, l -> new QoBOutputStreamManager(l, filename));
  }

  static synchronized void changeFileInAllAppenders(String newFilename) throws IOException {
    filename = newFilename;
    for (QoBOutputStreamManager manager : instances.values()) {
      manager.changeFile(newFilename);
    }
  }

  static synchronized void flushAllAppenders() {
    instances.values().forEach(OutputStreamManager::flush);
  }

  private static synchronized void remove(Layout<?> layout) {
    instances.remove(layout);
  }

  private final Layout<?> layout;
  private String currentFilename;

  private QoBOutputStreamManager(Layout<?> layout, String filename) {
    super(null, "QoBOutputStreamManager", layout, true);
    this.layout = layout;
    this.currentFilename = filename;
  }

  @Override
  protected OutputStream createOutputStream() throws IOException {
    Objects.requireNonNull(currentFilename, "no log file has been set");
    return new BufferedOutputStream(new FileOutputStream(currentFilename));
  }

  @Override
  public void close() {
    super.close();
    remove(layout);
  }

  private void changeFile(String newFilename) throws IOException {
    if (hasOutputStream()) {
      closeOutputStream();
    }
    currentFilename = newFilename;
    setOutputStream(createOutputStream());
  }
}
