package is.hail.nativecode;

import java.util.*;
import java.net.URL;
import com.sun.jna.*;


class NativeCode {
  static {
    System.err.println("NativeCode init ...");
    //
    // Access the library by the JNA which can find the
    // library bundled into a JAR file, but doesn't use
    // RTLD_GLOBAL to make symbols visible to everything.
    //
    NativeLibrary lib = NativeLibrary.getInstance("hail");
    String libName = lib.getFile().toString();
    System.err.println(libName);
    lib.dispose();
    //
    // Load the library again with RTLD_GLOBAL
    //
    System.load(libName);
    System.err.println("NativeCode init done");
  }

  final static String getIncludeDir() {
    String name = ClassLoader.getSystemResource("include/hail/hail.h").toString();
    int len = name.length();
    if (len >= 12) {
      name = name.substring(0, len-12);
    }
    if (name.substring(0, 5).equals("file:")) {
      name = name.substring(5, name.length());
    }
    return(name);
  }

  final static void forceLoad() { }
}
