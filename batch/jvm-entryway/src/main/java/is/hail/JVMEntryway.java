package is.hail;

import is.hail.QoBOutputStreamManager;
import java.io.*;
import java.lang.reflect.*;
import java.net.*;
import java.nio.*;
import java.nio.charset.*;
import java.util.*;
import java.util.concurrent.*;
import org.newsclub.net.unix.*;
import org.apache.logging.log4j.*;

class JVMEntryway {
  private static final Logger log = LogManager.getLogger(JVMEntryway.class); // this will initialize log4j which is required for us to access the QoBAppender in main
  private static final HashMap<String, ClassLoader> classLoaders = new HashMap<>();

  public static String throwableToString(Throwable t) throws IOException {
    try (StringWriter sw = new StringWriter();
         PrintWriter pw = new PrintWriter(sw)) {
      t.printStackTrace(pw);
      return sw.toString();
    }
  }

  private static int FINISH_USER_EXCEPTION = 0;
  private static int FINISH_ENTRYWAY_EXCEPTION = 1;
  private static int FINISH_NORMAL = 2;
  private static int FINISH_CANCELLED = 3;
  private static int FINISH_JVM_EOS = 4;  // NEVER USED ON JVM SIDE

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

        assert realArgs.length >= 4;
        String classPath = realArgs[0];
        String mainClass = realArgs[1];
        String scratchDir = realArgs[2];
        String logFile = realArgs[3];

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
        } else {
          System.err.println("reusing extant classLoader for " + classPath);
        }
        final ClassLoader hailRootCL = cl;

        Class<?> klass = hailRootCL.loadClass(mainClass);
        System.err.println("class loaded");
        Method main = klass.getDeclaredMethod("main", String[].class);
        System.err.println("main method got");

        QoBOutputStreamManager.instance().changeFile(logFile);
        log.info("is.hail.JVMEntryway received arguments:");
        for (int i = 0; i < nRealArgs; ++i) {
          log.info(i + ": " + realArgs[i]);
        }
        log.info("Yielding control to the QoB Job.");

        CompletionService<?> gather = new ExecutorCompletionService<Object>(executor);
        Future<?> mainThread = null;
        Future<?> shouldCancelThread = null;
        Future<?> completedThread = null;
        Throwable entrywayException = null;
        try {
          mainThread = gather.submit(new Runnable() {
              public void run() {
                ClassLoader oldClassLoader = Thread.currentThread().getContextClassLoader();
                Thread.currentThread().setContextClassLoader(hailRootCL);
                try {
                  String[] mainArgs = new String[nRealArgs - 2];
                  for (int i = 2; i < nRealArgs; ++i) {
                    mainArgs[i-2] = realArgs[i];
                  }
                  main.invoke(null, (Object) mainArgs);
                } catch (IllegalAccessException | InvocationTargetException e) {
                  log.error("QoB Job threw an exception.", e);
                  throw new RuntimeException(e);
                } catch (Exception e) {
                  log.error("QoB Job threw an exception.", e);
                } finally {
                  Thread.currentThread().setContextClassLoader(oldClassLoader);
                }
              }
            }, null);
          shouldCancelThread = gather.submit(new Runnable() {
              public void run() {
                ClassLoader oldClassLoader = Thread.currentThread().getContextClassLoader();
                Thread.currentThread().setContextClassLoader(hailRootCL);
                try {
                  int i = in.readInt();
                  assert i == 0 : i;
                } catch (IOException e) {
                  log.error("Exception encountered in QoB cancel thread.", e);
                  throw new RuntimeException(e);
                } catch (Exception e) {
		  log.error("Exception encountered in QoB cancel thread.", e);
                } finally {
                  Thread.currentThread().setContextClassLoader(oldClassLoader);
                }
              }
            }, null);
          completedThread = gather.take();
        } catch (Throwable t) {
          entrywayException = t;
        } finally {
	    QoBOutputStreamManager.instance().flush();
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
          assert(completedThread != null);

          if (completedThread == mainThread) {
            System.err.println("main thread done");
            finishFutures(out,
                          FINISH_NORMAL,
                          FINISH_USER_EXCEPTION,
                          mainThread,
                          FINISH_ENTRYWAY_EXCEPTION,
                          shouldCancelThread);
          } else {
            assert(completedThread == shouldCancelThread);
            System.err.println("cancelled");
            finishFutures(out,
                          FINISH_CANCELLED,
                          FINISH_ENTRYWAY_EXCEPTION,
                          shouldCancelThread,
                          FINISH_USER_EXCEPTION,
                          mainThread);
          }
        }
      }
      System.err.println("waiting for next connection");
      System.err.flush();
      System.out.flush();
    }
  }

  private static void finishFutures(DataOutputStream out,
                                    int finishedNormalType,
                                    int finishedExceptionType,
                                    Future finished,
                                    int secondaryExceptionType,
                                    Future secondary) throws IOException {
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

  private static void finishEntrywayException(DataOutputStream out, Throwable t) throws IOException {
    finishException(FINISH_ENTRYWAY_EXCEPTION, out, t);
  }

  private static void finishException(int type, DataOutputStream out, Throwable t) throws IOException {
    out.writeInt(type);
    String s = throwableToString(t);
    byte[] bytes = s.getBytes(StandardCharsets.UTF_8);
    out.writeInt(bytes.length);
    out.write(bytes);
  }

  private static Throwable cancelThreadRetrieveException(Future f) {
    f.cancel(true);
    return retrieveException(f);
  }

  private static Throwable retrieveException(Future f) {
    try {
      f.get();
    } catch (CancellationException e) {
    } catch (Throwable t) {
      return t;
    }
    return null;
  }
}

