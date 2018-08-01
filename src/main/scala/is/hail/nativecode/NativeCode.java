package is.hail.nativecode;

import java.io.*;
import java.util.*;
import java.util.regex.*;
import java.util.zip.*;
import java.net.URL;
import java.nio.file.*;
import com.sun.jna.*;
import org.apache.log4j.*;

public class NativeCode {
  private static String includeDir;
  
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
      includeDir = unpackHeadersToTmpIncludeDir();
    } catch (Throwable err) {
      System.err.println("FATAL: caught exception " + err.toString());
      err.printStackTrace();
      System.exit(1);
    }
  }
  
  private static Boolean isLinux() {
    String osName = System.getProperty("os.name").toLowerCase();
    if ((osName.length() >= 3) && osName.substring(0, 3).equals("mac")) {
      return false;
    }
    return true;
  }
  
  private static String libToLocalFile(String libName) throws IOException {
    File file = File.createTempFile(libName, ".so");
    ClassLoader loader = NativeCode.class.getClassLoader();
    InputStream s = null;
    if (isLinux()) {
      s = loader.getResourceAsStream("linux-x86-64/" + libName + ".so");
    } else {
      s = loader.getResourceAsStream("darwin/" + libName + ".dylib");
    }
    Files.copy(s, file.getAbsoluteFile().toPath(), StandardCopyOption.REPLACE_EXISTING);
    String path = file.getAbsoluteFile().toPath().toString();
    return path;
  }
    
  private static String unpackHeadersToTmpIncludeDir() {
    String result = "IncludeDirNotFound";
    String name = ClassLoader.getSystemResource("include").toString();
    if ((name.length() > 5) && name.substring(0, 5).equals("file:")) {
      // The resources are already unpacked
      result = name.substring(5, name.length());
    } else try {
      // The header files must be unpacked from a jar into local files
      int jarPos = name.indexOf("file:", 0);
      int jarEnd = name.indexOf("!", jarPos+1);
      String jarName = name.substring(jarPos+5, jarEnd);
      Path dirPath = Files.createTempDirectory("hail_headers_");
      File includeDir = new File(dirPath.toString());
      String tmpDirName = includeDir.getAbsolutePath().toString();
      result = tmpDirName + "/include";
      File f = new File(jarName);
      ZipFile zf = new ZipFile(f);
      Enumeration scan = zf.entries();
      while (scan.hasMoreElements()) {
        ZipEntry ze = (ZipEntry)scan.nextElement();
        String fileName = ze.getName();
        int len = fileName.length();
        if ((len > 8) &&
            fileName.substring(0, 8).equals("include/") &&
            fileName.substring(len-2, len).equals(".h")) {
          String dstName = tmpDirName + "/" + fileName;
          File dst = new File(dstName);
          Files.createDirectories(dst.toPath().getParent());
          InputStream src = zf.getInputStream(ze);
          Files.copy(src, dst.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
      }
    } catch (Throwable err) {
      System.err.println("FATAL: caught exception " + err.toString());
      err.printStackTrace();
      System.exit(1);
    }
    return result;
  }
  
  private native static long dlopenGlobal(String path);
  
  private native static long dlclose(long handle);

  public final static String getIncludeDir() {
    return includeDir;
  }

  public final static void forceLoad() { }
}
