package is.hail;

import java.io.*;
import java.lang.reflect.*;
import java.net.*;
import java.nio.charset.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.*;
import org.apache.logging.log4j.*;
import org.apache.logging.log4j.core.LoggerContext;
import org.newsclub.net.unix.*;

class JVMEntryway {
  // this will initialize log4j which is required for us to access the QoBAppender in main
  private static final Logger log = LogManager.getLogger(JVMEntryway.class);

  // spark classloaders are shared by all jobs that run against the same spark
  // version so that each JVM loads a given spark's classes (and extracts its
  // bundled native libraries) at most once
  private static final HashMap<String, ClassLoader> sparkClassLoaders = new HashMap<>();
  private static final HashMap<String, ClassLoader> jarClassLoaders = new HashMap<>();

  private static URL toUrl(File file) {
    try {
      return file.toURI().toURL();
    } catch (MalformedURLException e) {
      // unreachable: File.toURI() always yields an absolute file: URI
      throw new AssertionError(e);
    }
  }

  private static URL[] classPathToUrls(String classPath) {
    return Arrays.stream(classPath.split(","))
        .map(File::new)
        .flatMap(file -> file.isDirectory() ? Arrays.stream(file.listFiles()) : Stream.of(file))
        .map(JVMEntryway::toUrl)
        .toArray(URL[]::new);
  }

  private static ClassLoader classLoaderFor(String sparkClassPath, String jarClassPath) {
    return jarClassLoaders.computeIfAbsent(sparkClassPath + "," + jarClassPath, key -> {
      System.err.println("creating classLoader for " + key);
      var sparkCl = sparkClassLoaders.computeIfAbsent(sparkClassPath, sparkKey -> {
        System.err.println("creating spark classLoader for " + sparkKey);
        return new URLClassLoader(classPathToUrls(sparkKey));
      });
      return new URLClassLoader(classPathToUrls(jarClassPath), sparkCl);
    });
  }

  public static String throwableToString(Throwable t) {
    var sw = new StringWriter();
    try (var pw = new PrintWriter(sw)) {
      t.printStackTrace(pw);
      return sw.toString();
    }
  }

  private static final int FINISH_USER_EXCEPTION = 0;
  private static final int FINISH_ENTRYWAY_EXCEPTION = 1;
  private static final int FINISH_NORMAL = 2;
  private static final int FINISH_CANCELLED = 3;
  private static final int FINISH_JVM_EOS = 4; // NEVER USED ON JVM SIDE

  public static void main(String[] args) throws Exception {
    assert args.length == 1;
    AFUNIXServerSocket server = AFUNIXServerSocket.newInstance();
    server.bind(AFUNIXSocketAddress.of(new File(args[0])));
    System.err.println("listening on " + args[0]);
    try (AFUNIXSocket socket = server.accept()) {
      System.err.println("negotiating start up with worker");
      DataInputStream in = new DataInputStream(socket.getInputStream());
      DataOutputStream out = new DataOutputStream(socket.getOutputStream());
      System.err.flush();
      out.writeBoolean(true);
      assert (in.readBoolean());
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

        assert realArgs.length >= 5;
        var sparkClassPath = realArgs[0];
        var jarClassPath = realArgs[1];
        var mainClass = realArgs[2];
        var logFile = realArgs[4];

        final var hailRootCL = classLoaderFor(sparkClassPath, jarClassPath);

        Class<?> klass = hailRootCL.loadClass(mainClass);
        System.err.println("class loaded");
        Method main = klass.getDeclaredMethod("main", String[].class);
        System.err.println("main method got");

        QoBOutputStreamManager.changeFileInAllAppenders(logFile);
        log.info("is.hail.JVMEntryway received arguments:");
        for (int i = 0; i < nRealArgs; ++i) {
          log.info("{}: {}", i, realArgs[i]);
        }
        log.info("Yielding control to the QoB Job.");

        CompletionService<?> gather = new ExecutorCompletionService<Object>(executor);
        Future<?> mainThread = null;
        Future<?> shouldCancelThread = null;
        Future<?> completedThread = null;
        Throwable entrywayException = null;
        try {
          mainThread = gather.submit(
              new Runnable() {
                public void run() {
                  ClassLoader oldClassLoader = Thread.currentThread().getContextClassLoader();
                  Thread.currentThread().setContextClassLoader(hailRootCL);
                  try {
                    String[] mainArgs = new String[nRealArgs - 3];
                    for (int i = 3; i < nRealArgs; ++i) {
                      mainArgs[i - 3] = realArgs[i];
                    }
                    main.invoke(null, (Object) mainArgs);
                  } catch (IllegalAccessException | InvocationTargetException e) {
                    log.error("QoB Job threw an exception.", e);
                    throw new RuntimeException(e);
                  } catch (Exception e) {
                    log.error("QoB Job threw an exception.", e);
                  } finally {
                    QoBOutputStreamManager.flushAllAppenders();
                    Thread.currentThread().setContextClassLoader(oldClassLoader);
                  }
                }
              },
              null);
          shouldCancelThread = gather.submit(
              new Runnable() {
                public void run() {
                  ClassLoader oldClassLoader = Thread.currentThread().getContextClassLoader();
                  Thread.currentThread().setContextClassLoader(hailRootCL);
                  try {
                    int i = in.readInt();
                    assert i == 0 : i;
                  } catch (EOFException e) {
                    // the worker closes the socket without writing a cancel
                    // signal at the end of every non-cancelled job
                    log.info("worker closed the connection without requesting cancellation");
                    throw new RuntimeException(e);
                  } catch (IOException e) {
                    log.error("Exception encountered in QoB cancel thread.", e);
                    throw new RuntimeException(e);
                  } catch (Exception e) {
                    log.error("Exception encountered in QoB cancel thread.", e);
                  } finally {
                    QoBOutputStreamManager.flushAllAppenders();
                    Thread.currentThread().setContextClassLoader(oldClassLoader);
                  }
                }
              },
              null);
          completedThread = gather.take();
        } catch (Throwable t) {
          entrywayException = t;
        }

        if (entrywayException != null) {
          System.err.println("exception in entryway code");
          entrywayException.printStackTrace();

          if (mainThread != null) {
            Throwable t2 = cancelThreadRetrieveException(mainThread);
            if (t2 != null) {
              entrywayException.addSuppressed(t2);
            }
          }

          if (shouldCancelThread != null) {
            Throwable t2 = cancelThreadRetrieveException(shouldCancelThread);
            if (t2 != null) {
              entrywayException.addSuppressed(t2);
            }
          }

          finishEntrywayException(out, entrywayException);
        } else {
          assert (completedThread != null);

          if (completedThread == mainThread) {
            System.err.println("main thread done");
            finishFutures(
                out,
                FINISH_NORMAL,
                FINISH_USER_EXCEPTION,
                mainThread,
                FINISH_ENTRYWAY_EXCEPTION,
                shouldCancelThread);
          } else {
            assert (completedThread == shouldCancelThread);
            System.err.println("cancelled");
            finishFutures(
                out,
                FINISH_CANCELLED,
                FINISH_ENTRYWAY_EXCEPTION,
                shouldCancelThread,
                FINISH_USER_EXCEPTION,
                mainThread);
          }
        }
      } finally {
        QoBOutputStreamManager.flushAllAppenders();
        LoggerContext context = (LoggerContext) LogManager.getContext(false);
        ClassLoader loader = JVMEntryway.class.getClassLoader();
        URL url = loader.getResource("log4j2.properties");
        System.err.println("reconfiguring logging " + url.toString());
        context.setConfigLocation(url.toURI()); // this will force a reconfiguration
      }
      System.err.println("waiting for next connection");
      System.err.flush();
      System.out.flush();
    }
  }

  private static void finishFutures(
      DataOutputStream out,
      int finishedNormalType,
      int finishedExceptionType,
      Future<?> finished,
      int secondaryExceptionType,
      Future<?> secondary)
      throws IOException {
    Throwable finishedException = retrieveException(finished);
    Throwable secondaryException = cancelThreadRetrieveException(secondary);

    if (finishedException != null) {
      if (secondaryException != null) {
        finishedException.addSuppressed(secondaryException);
      }
      finishException(finishedExceptionType, out, finishedException);
    } else if (secondaryException != null) {
      finishException(secondaryExceptionType, out, secondaryException);
    } else {
      out.writeInt(finishedNormalType);
    }
  }

  private static void finishUserException(DataOutputStream out, Throwable t) throws IOException {
    finishException(FINISH_USER_EXCEPTION, out, t);
  }

  private static void finishEntrywayException(DataOutputStream out, Throwable t)
      throws IOException {
    finishException(FINISH_ENTRYWAY_EXCEPTION, out, t);
  }

  private static void finishException(int type, DataOutputStream out, Throwable t)
      throws IOException {
    out.writeInt(type);
    String s = throwableToString(t);
    byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
    out.writeInt(bytes.length);
    out.write(bytes);
  }

  private static Throwable cancelThreadRetrieveException(Future<?> f) {
    f.cancel(true);
    return retrieveException(f);
  }

  private static Throwable retrieveException(Future<?> f) {
    try {
      f.get();
    } catch (CancellationException ignored) {
    } catch (Throwable t) {
      return t;
    }
    return null;
  }
}
