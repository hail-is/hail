package is.hail

import javax.net.ssl.SSLException
import java.net._
import java.io.EOFException
import java.util.concurrent.TimeoutException
import is.hail.utils._

import org.apache.http.NoHttpResponseException
import org.apache.http.ConnectionClosedException
import org.apache.http.conn.HttpHostConnectException
import org.apache.log4j.{LogManager, Logger}

import is.hail.shadedazure.reactor.core.Exceptions.ReactiveException
import is.hail.shadedazure.com.azure.storage.common.implementation.Constants
import scala.util.Random
import java.io._
import com.google.cloud.storage.StorageException
import com.google.api.client.googleapis.json.GoogleJsonResponseException
import com.google.api.client.http.HttpResponseException

package object services {
  lazy val log: Logger = LogManager.getLogger("is.hail.services")

  val RETRYABLE_HTTP_STATUS_CODES: Set[Int] = {
    val s = Set(408, 429, 500, 502, 503, 504)
    if (System.getenv("HAIL_DONT_RETRY_500") == "1")
      s - 500
    else
      s
  }

  private[this] val LOG_2_MAX_MULTIPLIER = 30  // do not set larger than 30 due to integer overflow calculating multiplier
  private[this] val DEFAULT_MAX_DELAY_MS = 60000
  private[this] val DEFAULT_BASE_DELAY_MS = 1000

  def delayMsForTry(
    tries: Int,
    baseDelayMs: Int = DEFAULT_BASE_DELAY_MS,
    maxDelayMs: Int = DEFAULT_MAX_DELAY_MS
  ): Int = {
    // Based on AWS' recommendations:
    // - https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    // - https://github.com/aws/aws-sdk-java/blob/master/aws-java-sdk-core/src/main/java/com/amazonaws/retry/PredefinedBackoffStrategies.java
    val multiplier = 1L << math.min(tries, LOG_2_MAX_MULTIPLIER)
    val ceilingForDelayMs = math.min(baseDelayMs * multiplier, maxDelayMs).toInt
    val proposedDelayMs = ceilingForDelayMs / 2 + Random.nextInt(ceilingForDelayMs / 2 + 1)
    return proposedDelayMs
  }

  def sleepBeforTry(
    tries: Int,
    baseDelayMs: Int = DEFAULT_BASE_DELAY_MS,
    maxDelayMs: Int = DEFAULT_MAX_DELAY_MS
  ) = {
    Thread.sleep(delayMsForTry(tries, baseDelayMs, maxDelayMs))
  }

  def isLimitedRetriesError(_e: Throwable): Boolean = {
    // An exception is a "retry once error" if a rare, known bug in a dependency or in a cloud
    // provider can manifest as this exception *and* that manifestation is indistinguishable from a
    // true error.
    val e = is.hail.shadedazure.reactor.core.Exceptions.unwrap(_e)
    e match {
      case e: RuntimeException
          if e.getMessage != null && e.getMessage == Constants.STREAM_CLOSED =>
        true
      case e: SocketException
          if e.getMessage != null && e.getMessage.contains("Connection reset") =>
        true
      case e: HttpResponseException
          if e.getStatusCode() == 400 && e.getMessage != null && (
            e.getMessage.contains("Invalid grant: account not found") ||
              e.getMessage.contains("{\"error\":\"unhandled_canonical_code_14\"}")
          ) =>
        true
      case e: IOException
          if e.getMessage != null && e.getMessage.contains("Connection reset by peer") =>
	// java.io.IOException: Connection reset by peer
	//   at sun.nio.ch.FileDispatcherImpl.read0(NativeMethod) ~[?:1.8.0_362]
	//   at sun.nio.ch.SocketDispatcher.read(SocketDispatcher.java:39)~[?:1.8.0_362]
	//   at sun.nio.ch.IOUtil.readIntoNativeBuffer(IOUtil.java:223)~[?:1.8.0_362]
	//   at sun.nio.ch.IOUtil.read(IOUtil.java:192) ~[?:1.8.0_362]
	//   at sun.nio.ch.SocketChannelImpl.read(SocketChannelImpl.java:379) ~[?:1.8.0_362]
        true
      case e =>
        val cause = e.getCause
        cause != null && isLimitedRetriesError(cause)
    }
  }

  def isTransientError(_e: Throwable): Boolean = {
    // ReactiveException is package private inside reactore.core.Exception so we cannot access
    // it directly for an isInstance check. AFAICT, this is the only way to check if we received
    // a ReactiveException.
    //
    // If the argument is a ReactiveException, it returns its cause. If the argument is not a
    // ReactiveException it returns the exception unmodified.
    val e = is.hail.shadedazure.reactor.core.Exceptions.unwrap(_e)
    e match {
      case e: NoHttpResponseException =>
        true
      case e: HttpResponseException
          if RETRYABLE_HTTP_STATUS_CODES.contains(e.getStatusCode()) =>
        true
      case e: HttpResponseException
          if (e.getStatusCode() == 410 &&
            e.getMessage != null &&
            e.getMessage.contains("\"code\": 503,") &&
            e.getMessage.contains("\"message\": \"Backend Error\",")
          ) =>
        // hail.utils.java.FatalError: HttpResponseException: 410 Gone
        // PUT https://storage.googleapis.com/upload/storage/v1/b/hail-test-ezlis/o?name=tmp/hail/nBHPQsrxGvJ4T7Ybdp1IjQ/persist_TableObF6TwC6hv/rows/metadata.json.gz&uploadType=resumable&upload_id=ADPycdsFEtq65NC-ahk6tt6qdD3bKC3asqVSJELnirlpLG_ZDV_637Nn7NourXYTgMRKlX3bQVe9BfD_QfIP_kupTxVQyrJWQJrj
        // {
        //   "error": {
        //     "code": 503,
        //     "message": "Backend Error",
        //     "errors": [
        //       {
        //         "message": "Backend Error",
        //         "domain": "global",
        //         "reason": "backendError"
        //       }
        //     ]
        //   }
        // }
        true
      case e: ClientResponseException
          if RETRYABLE_HTTP_STATUS_CODES.contains(e.status) =>
        true
      case e: GoogleJsonResponseException
          if RETRYABLE_HTTP_STATUS_CODES.contains(e.getStatusCode()) =>
        true
      case e: HttpHostConnectException =>
        true
      case e: NoRouteToHostException =>
        true
      case e: SocketTimeoutException =>
        true
      case e: java.util.concurrent.TimeoutException =>
        true
      case e: UnknownHostException =>
        true
      case e: ConnectionClosedException =>
        true
      case e: SocketException
          if e.getMessage != null && (
            e.getMessage.contains("Connection timed out (Read failed)") ||
              e.getMessage.contains("Broken pipe") ||
              e.getMessage.contains("Connection refused")) =>
        true
      case e: EOFException
          if e.getMessage != null && e.getMessage.contains("SSL peer shut down incorrectly") =>
        true
      case e: IllegalStateException
          if e.getMessage.contains("Timeout on blocking read") =>
        // Caused by: java.lang.IllegalStateException: Timeout on blocking read for 30000000000 NANOSECONDS
        // reactor.core.publisher.BlockingSingleSubscriber.blockingGet(BlockingSingleSubscriber.java:123)
        // reactor.core.publisher.Mono.block(Mono.java:1727)
        // com.azure.storage.common.implementation.StorageImplUtils.blockWithOptionalTimeout(StorageImplUtils.java:130)
        // com.azure.storage.blob.specialized.BlobClientBase.downloadStreamWithResponse(BlobClientBase.java:731)
        // is.hail.io.fs.AzureStorageFS$$anon$1.fill(AzureStorageFS.scala:152)
        // is.hail.io.fs.FSSeekableInputStream.read(FS.scala:141)
        // ...
        true
      case e: java.net.SocketTimeoutException
          if e.getMessage != null && e.getMessage.contains("connect timed out") =>
        true
      case e @ (_: SSLException | _: StorageException | _: IOException)
          if e.getCause != null && NettyProxy.isRetryableNettyIOException(e.getCause) =>
        true
      case e =>
        val cause = e.getCause
        cause != null && isTransientError(cause)
    }
  }

  def retryTransientErrors[T](f: => T, reset: Option[() => Unit] = None): T = {
    var tries = 0
    while (true) {
      try {
        return f
      } catch {
        case e: Exception =>
          tries += 1
          val delay = delayMsForTry(tries)
          if (tries <= 5 && isLimitedRetriesError(e)) {
            log.warn(
              s"A limited retry error has occured. We will automatically retry " +
                s"${5 - tries} more times. Do not be alarmed. (next delay: " +
                s"$delay). The most recent error was $e.")
          } else if (!isTransientError(e)) {
            throw e
          } else if (tries % 10 == 0) {
            log.warn(s"Encountered $tries transient errors, most recent one was $e.")
          }
          Thread.sleep(delay)
      }
      reset.foreach(_())
    }

    throw new AssertionError("unreachable")
  }

  def formatException(e: Throwable): String = {
    using(new StringWriter()) { sw =>
      using(new PrintWriter(sw)) { pw =>
        e.printStackTrace(pw)
        sw.toString
      }
    }
  }
}
