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
  private static String hailName;
  
  static {
    hailName = "hail_abi_v9";
    try {
      // libboot.so has native methods to call dlopen/dlclose
      String libBoot = libToLocalFile("libboot");
      System.load(libBoot);
      if (isLinux()) {
        // libhail_abi_v9 works with libstdc++ 6.0.21 (g++-5.x) and later.
        // libhail_abi_v2 works with systems based on g++-3.4.x to g++-4.9.x
        String cxx = System.getenv("CXX");
        if (cxx == null) cxx = "c++";
        String version = null;
        try {
          Process child = Runtime.getRuntime().exec(cxx + " --version");
          BufferedReader s = new BufferedReader(new InputStreamReader(child.getInputStream()));
          version = s.readLine();
          child.waitFor();
        } catch (Throwable err) {
          // Couldn't run c++
        }
        if (version != null) {
          boolean isClang = (version.indexOf("clang version", 0) >= 0);
          int idx = version.indexOf(".", 0);
          if (idx > 0) {
            int j = idx-1;
            while ((j >= 0) && ('0' <= version.charAt(j)) && (version.charAt(j) <= '9')) j -= 1;
            int major = Integer.parseInt(version.substring(j+1, idx));
            if (isClang) {
              // We'll try to be conservative for clang-3.x versions, from the really
              // early days of C++11.  But it's much more likely it will be clang-4.x
              // or later which will use the newer abi_v9.
              if (major <= 3) hailName = "hail_abi_v2";
            } else {
              // Use abi_v2 for g++-3.4.0 to g++-4.9.x
              if (major <= 4) hailName = "hail_abi_v2";
            }
          }
        }
      } else {
        // Since MacOS 10.9 Mavericks (Oct 2013) the default library is libc++
        // rather than libstdc++.  I believe all versions of libc++ support
        // at least abi_v9.
        hailName = "hail_abi_v9";
      } 
      String libHail = libToLocalFile("lib"+hailName);
      // We need libhail.so to be loaded with RTLD_GLOBAL so that its symbols
      // are visible to dynamic-generated code in other DLLs ...
      long handle = dlopenGlobal(libHail);
      // ... but we also need System.load to let the JVM see it
      System.load(libHail);
      includeDir = unpackHeadersToTmpIncludeDir();
    } catch (Throwable err) {
      System.err.println("FATAL: caught exception " + err.toString());
      err.printStackTrace();
      System.exit(1);
    }
  }

  // Scala classes using jna and Native.register(libname) need to use
  // this to get either "hail_abi_v2" or "hail_abi_v9"
  public static String getHailName() {
    return hailName;
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
