package is.hail.nativecode;

import java.io.*;

class NativeCode {
  static {
    try {
      File file = File.createTempFile("libhail", ".lib");
      ClassLoader loader = NativeCode.class.getClassLoader();
      InputStream s = null;
      String osName = System.getProperty("os.name");
      if (osName.equals("Linux") || osName.equals("linux")) {
        s = loader.getResourceAsStream("linux-x86-64/libhail.so");
      } else {
        s = loader.getResourceAsStream("darwin/libhail.dylib");
      }
      java.nio.file.Files.copy(s, file.getAbsoluteFile().toPath(),
        java.nio.file.StandardCopyOption.REPLACE_EXISTING
      );
      String path = file.getAbsoluteFile().toPath().toString();
      // Unlike JNA, this loads it with RTLD_GLOBAL to make all symbols
      // visible to DLL's loaded in future
      System.load(path);
    } catch (Exception e) {
      System.err.println("ERROR: NativeCode.init caught exception");
    }
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
