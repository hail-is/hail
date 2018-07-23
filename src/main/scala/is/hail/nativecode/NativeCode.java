package is.hail.nativecode;

import java.io.*;
import java.util.*;
import java.net.URL;
import com.sun.jna.*;
import org.apache.log4j.*;

public class NativeCode {
  static {
    try {
      String libHail = libToLocalFile("libhail");
      if (isLinux()) {
        // libboot.so has native methods to call dlopen/dlclose
        String libBoot = libToLocalFile("libboot");
        System.load(libBoot);
        // We need libhail.so to be loaded with RTLD_GLOBAL so that its symbols
        // are visible to dynamic-generated code in other DLLs ...
        long handle = dlopenGlobal(libHail);
        // ... but we also need System.load to let the JVM see it
        System.load(libHail);
      } else {
        // MacOS System.load uses RTLD_GLOBAL, so it just works
        System.load(libHail);
      }
    } catch (Exception e) {
      handleException(e);
    }
  }
  
  private static void handleException(Exception e) {
    String msg = "NativeCode.init caught exception: " + e.toString();
    LogManager.getLogger("Hail").error(msg);
    LogManager.getRootLogger().error(msg);
  }
  
  private static Boolean isLinux() {
    String osName = System.getProperty("os.name").toLowerCase();
    if ((osName.length() >= 3) && osName.substring(0, 3).equals("mac")) {
      return false;
    }
    return true;
  }
  
  private static String libToLocalFile(String libName) {
    String path = "";
    try {
      File file = File.createTempFile(libName, ".so");
      ClassLoader loader = NativeCode.class.getClassLoader();
      InputStream s = null;
      if (isLinux()) {
        s = loader.getResourceAsStream("linux-x86-64/" + libName + ".so");
      } else {
        s = loader.getResourceAsStream("darwin/" + libName + ".dylib");
      }
      java.nio.file.Files.copy(s, file.getAbsoluteFile().toPath(),
        java.nio.file.StandardCopyOption.REPLACE_EXISTING
      );
      path = file.getAbsoluteFile().toPath().toString();
    } catch (Exception e) {
      handleException(e);
      path = libName + "_resource_not_found";
    }
    return path;
  }
  
  private native static long dlopenGlobal(String path);
  
  private native static long dlclose(long handle);

  public final static String getIncludeDir() {
    String name = ClassLoader.getSystemResource("include/hail/hail.h").toString();
    int len = name.length();
    if (len >= 12) {
      name = name.substring(0, len-12);
    }
    if (name.substring(0, 5).equals("file:")) {
      name = name.substring(5, name.length());
    }
    return name;
  }

  public final static void forceLoad() { }
}
