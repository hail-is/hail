package is.hail.nativecode;

import org.apache.spark.TaskContext;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Enumeration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

public class NativeCode {
    private static String includeDir;
    private static String hailName;

    static {
        hailName = "hail";
        try {
            // libboot.so has native methods to call dlopen/dlclose
            String libBoot = libToLocalFile("libboot");
            System.load(libBoot);
            String libHail = libToLocalFile("lib" + hailName);
            // We need libhail.so to be loaded with RTLD_GLOBAL so that its symbols
            // are visible to dynamic-generated code in other DLLs ...
            long handle = dlopenGlobal(libHail);
            // ... but we also need System.load to let the JVM see it
            System.load(libHail);
        } catch (Throwable err) {
            System.err.println("FATAL: caught exception " + err.toString());
            err.printStackTrace();
            System.exit(1);
        }
    }

    public static String getHailName() {
        // Hail methods implemented with JNA call Native.register(NativeCode.getHailName()),
        // so that NativeCode gets to choose which DLL to load (though currently there's
        // only one), *and* to get it loaded with RTLD_GLOBAL so that it also works for
        // dynamic-generated C++ code.
        return hailName;
    }

    private static Boolean isMac() {
        String osName = System.getProperty("os.name").toLowerCase();
        return osName.length() >= 3 && osName.substring(0, 3).equals("mac");
    }

    private static String libToLocalFile(String libName) throws IOException {
        File file = File.createTempFile(libName, ".so");
        ClassLoader loader = NativeCode.class.getClassLoader();
        String resourceName = isMac() 
            ? "darwin/" + libName + ".dylib" 
            : "linux-x86-64/" + libName + ".so";
        InputStream s = loader.getResourceAsStream(resourceName);
        if (s == null) {
            throw new RuntimeException(
                "Native library " + resourceName + " could not be found, does the JAR contain this resource?");
        }
        Files.copy(s, file.getAbsoluteFile().toPath(), StandardCopyOption.REPLACE_EXISTING);
        return file.getAbsoluteFile().toPath().toString();
    }

    private static String unpackHeadersToTmpIncludeDir() {
        String result = null;

        try {
            Class<?> nativeCodeClass = Class.forName("is.hail.nativecode.NativeCode");
            String name = nativeCodeClass.getClassLoader().getResource("include").toString();
            if ((name.length() > 5) && name.substring(0, 5).equals("file:")) {
                // The resources are already unpacked
                result = name.substring(5, name.length());
            } else {
                // The header files must be unpacked from a jar into local files
                int jarPos = name.indexOf("file:", 0);
                int jarEnd = name.indexOf("!", jarPos + 1);
                String jarName = name.substring(jarPos + 5, jarEnd);
                Path dirPath = Files.createTempDirectory("hail_headers_");
                File includeDir = new File(dirPath.toString());
                String tmpDirName = includeDir.getAbsolutePath();
                result = tmpDirName + "/include";
                File f = new File(jarName);
                ZipFile zf = new ZipFile(f);
                Enumeration<? extends ZipEntry> scan = zf.entries();
                while (scan.hasMoreElements()) {
                    ZipEntry ze = scan.nextElement();
                    String fileName = ze.getName();
                    int len = fileName.length();
                    if ((len > 8) &&
                            fileName.substring(0, 8).equals("include/") &&
                            fileName.substring(len - 2, len).equals(".h")) {
                        String dstName = tmpDirName + "/" + fileName;
                        File dst = new File(dstName);
                        Files.createDirectories(dst.toPath().getParent());
                        InputStream src = zf.getInputStream(ze);
                        Files.copy(src, dst.toPath(), StandardCopyOption.REPLACE_EXISTING);
                    }
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

    public synchronized static String getIncludeDir() {
        assert (TaskContext.get() == null);
        if (includeDir == null)
            includeDir = unpackHeadersToTmpIncludeDir();
        return includeDir;
    }

    public static void forceLoad() {
    }
}
