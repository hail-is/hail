package is.hail.nativecode;

import java.util.*;
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

  final static void forceLoad() { }
}
