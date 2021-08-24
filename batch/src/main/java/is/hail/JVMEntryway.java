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
  // private static final HashMap<String, ClassLoader> classLoaders = new HashMap<>();

  public static String throwableToString(Throwable t) throws IOException {
    try (StringWriter sw = new StringWriter();
         PrintWriter pw = new PrintWriter(sw)) {
      t.printStackTrace(pw);
      return sw.toString();
    }
  }

  private static int FINISH_EXCEPTION = 0;
  private static int FINISH_NORMAL = 1;
  private static int FINISH_CANCELLED = 2;

  public static void main(String[] args) throws Exception {
    assert args.length == 1;
    AFUNIXServerSocket server = AFUNIXServerSocket.newInstance();
    server.bind(new AFUNIXSocketAddress(new File(args[0])));
    System.err.println("listening on " + args[0]);
    try (AFUNIXSocket socket = server.accept()) {
      System.err.println("negotiating start up with worker");
      DataInputStream in = new DataInputStream(socket.getInputStream());
      DataOutputStream out = new DataOutputStream(socket.getOutputStream());
      System.err.flush();
      out.writeBoolean(true);
      assert(in.readBoolean());
    }
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
        ClassLoader cl = new URLClassLoader(urls.toArray(new URL[0]));

        // if (hailRootCL == null) {
        //   System.err.println("no extant classLoader for " + classPath);
        //   String[] urlStrings = classPath.split(",");
        //   ArrayList<URL> urls = new ArrayList<>();
        //   for (int i = 0; i < urlStrings.length; ++i) {
        //     File file = new File(urlStrings[i]);
        //     urls.add(file.toURI().toURL());
        //     if (file.isDirectory()) {
        //       for (final File f : file.listFiles()) {
        //         urls.add(f.toURI().toURL());
        //       }
        //     }
        //   }
        //   hailRootCL = new URLClassLoader(urls.toArray(new URL[0]));
        //   classLoaders.put(classPath, hailRootCL);
        // }
        // System.err.println("have classLoader for " + classPath);
        // URL[] emptyURLArray = {};
        // ClassLoader cl = new URLClassLoader(emptyURLArray, hailRootCL);
        System.err.println("have fresh classLoader for this job");
        Class<?> klass = cl.loadClass(mainClass);
        System.err.println("class loaded ");
        Method main = klass.getDeclaredMethod("main", String[].class);
        System.err.println("main method got ");

        CompletionService<?> gather = new ExecutorCompletionService<Object>(executor);
        Future<?> mainThread = null;
        Future<?> shouldCancelThread = null;
        Future<?> completedThread = null;
        Throwable exception = null;
        try {
          System.err.println("submitting main ");
          mainThread = gather.submit(new Runnable() {
              public void run() {
                ClassLoader oldClassLoader = Thread.currentThread().getContextClassLoader();
                Thread.currentThread().setContextClassLoader(cl);
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
                } finally {
                  Thread.currentThread().setContextClassLoader(oldClassLoader);
                }
              }
            }, null);
          System.err.println("submitting shouldCancel");
          shouldCancelThread = gather.submit(new Runnable() {
              public void run() {
                ClassLoader oldClassLoader = Thread.currentThread().getContextClassLoader();
                Thread.currentThread().setContextClassLoader(cl);
                try {
                  System.err.println("shouldCancel running");
                  int i = in.readInt();
                  System.err.println("will cancel");
                  assert i == 0 : i;
                } catch (IOException e) {
                  throw new RuntimeException(e);
                } finally {
                  Thread.currentThread().setContextClassLoader(oldClassLoader);
                }
              }
            }, null);
          System.err.println("waiting");
          completedThread = gather.take();
        } catch (Throwable t) {
          exception = t;
        }

        if (exception != null) {
          System.err.println("Encountered exception during execution");
          exception.printStackTrace();

          mainThread.cancel(true);
          shouldCancelThread.cancel(true);

          mainThread.get();
          shouldCancelThread.get();

          out.writeInt(FINISH_EXCEPTION);
          String s = throwableToString(exception);
          byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
          out.writeInt(bytes.length);
          out.write(bytes);
        } else {
          assert(completedThread != null);
          if (completedThread == mainThread) {
            System.err.println("main thread done");
            shouldCancelThread.cancel(true);

            mainThread.get();  // retrieve any exceptions
            try {
              shouldCancelThread.get();  // wait for the thread to cancel and retrieve exceptions
            } catch (CancellationException e) {
            }

            out.writeInt(FINISH_NORMAL);
          } else {
            System.err.println("I was cancelled");
            assert(completedThread == shouldCancelThread);
            mainThread.cancel(true);

            shouldCancelThread.get();  // retrieve any exceptions
            try {
              mainThread.get();  // wait for the thread to cancel and retrieve exceptions
            } catch (CancellationException e) {
            }

            out.writeInt(FINISH_CANCELLED);
            out.flush();
          }
        }
      }
      System.err.println("waiting for next connection");
      System.err.flush();
    }
  }
}

