package is.hail;

import java.io.*;
import java.lang.reflect.*;
import java.net.*;
import java.nio.*;
import java.nio.charset.*;
import java.util.*;
import java.util.concurrent.*;
import org.newsclub.net.unix.*;

class JVMEntryway {
  private static final HashMap<String, ClassLoader> classLoaders = new HashMap<>();

  public static String throwableToString(Throwable t) throws IOException {
    try (StringWriter sw = new StringWriter();
         PrintWriter pw = new PrintWriter(sw)) {
      t.printStackTrace(pw);
      return sw.toString();
    }
  }

  public static void main(String[] args) throws Exception {
    assert args.length == 1;
    AFUNIXServerSocket server = AFUNIXServerSocket.newInstance();
    server.bind(new AFUNIXSocketAddress(new File(args[0])));
    System.err.println("listening on " + args[0]);
    ExecutorService executor = Executors.newFixedThreadPool(2);
    while (true) {
      try (AFUNIXSocket socket = server.accept()) {
        System.err.println("connection accepted");
        DataInputStream in = new DataInputStream(socket.getInputStream());
        DataOutputStream out = new DataOutputStream(socket.getOutputStream());
        int nRealArgs = in.readInt();
        System.err.println("reading " + nRealArgs + " arguments");
        String[] realArgs = new String[nRealArgs];
        for (int i = 0; i < nRealArgs; ++i) {
          int length = in.readInt();
          byte[] bytes = new byte[length];
          System.err.println("reading " + i + ": length=" + length);
          in.read(bytes);
          realArgs[i] = new String(bytes);
          System.err.println("reading " + i + ": " + realArgs[i]);
        }

        assert realArgs.length >= 2;
        String classPath = realArgs[0];
        String mainClass = realArgs[1];
        ClassLoader cl = classLoaders.get(classPath);
        if (cl == null) {
          System.err.println("no extant classLoader for " + classPath);
          String[] urlStrings = classPath.split(",");
          ArrayList<URL> urls = new ArrayList<>();
          for (int i = 0; i < urlStrings.length; ++i) {
            File file = new File(urlStrings[i]);
            urls.add(file.toURI().toURL());
            if (file.isDirectory()) {
              for (final File f : file.listFiles()) {
                urls.add(f.toURI().toURL());
              }
            }
          }
          cl = new URLClassLoader(urls.toArray(new URL[0]));
          classLoaders.put(classPath, cl);
        }
        System.err.println("have classLoader for " + classPath);
        Class<?> klass = cl.loadClass(mainClass);
        System.err.println("class loaded ");
        Method main = klass.getDeclaredMethod("main", String[].class);
        System.err.println("main method got ");
        ClassLoader oldClassLoader = Thread.currentThread().getContextClassLoader();
        Thread.currentThread().setContextClassLoader(cl);
        System.err.println("context class loader set ");

        Boolean invoked = false;
        CompletionService<?> gather = new ExecutorCompletionService<Object>(executor);
        try {
          System.err.println("submitting main ");
          Future<?> mainThread = gather.submit(new Runnable() {
              public void run() {
                try {
                  System.err.println("main running");
                  String[] mainArgs = new String[nRealArgs - 2];
                  for (int i = 2; i < nRealArgs; ++i) {
                    mainArgs[i-2] = realArgs[i];
                  }
                  System.err.println("invoking");
                  main.invoke(null, (Object) mainArgs);
                  System.err.println("done invoking");
                } catch (IllegalAccessException | InvocationTargetException e) {
                  throw new RuntimeException(e);
                }
              }
            }, null);
          System.err.println("submitting shouldCancel");
          Future<?> shouldCancelThread = gather.submit(new Runnable() {
              public void run() {
                try {
                  System.err.println("shouldCancel running");
                  int i = in.readInt();
                  System.err.println("will cancel");
                  assert i == 0 : i;
                } catch (IOException e) {
                  throw new RuntimeException(e);
                }
              }
            }, null);
          System.err.println("waiting");
          Future<?> completedThread = gather.take();
          if (completedThread == mainThread) {
            System.err.println("main thread done");
            shouldCancelThread.cancel(true);
            mainThread.get();
            invoked = true;
            out.writeBoolean(true);
          } else {
            System.err.println("I was cancelled");
            assert completedThread == shouldCancelThread;
            mainThread.cancel(true);
            shouldCancelThread.get();
            out.writeInt(0);
            out.flush();
          }
        } catch (Throwable t) {
          if (invoked) {
            throw t;
          }

          out.writeBoolean(false);
          String s = throwableToString(t);
          byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
          out.writeInt(bytes.length);
          out.write(bytes);
        }
        Thread.currentThread().setContextClassLoader(oldClassLoader);
      }
      System.err.println("waiting for next connection");
    }
  }
}

