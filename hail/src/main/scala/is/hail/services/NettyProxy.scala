package is.hail.services

import io.netty.channel.epoll.Epoll
import io.netty.channel.unix.Errors // cannot be in package.scala because is.hail.io shadows top-level io

object NettyProxy {
  val isRetryableNettyIOException: Throwable => Boolean = if (Epoll.isAvailable) {
    // Epoll.isAvailable returns true iff the io.netty.channel.unix.Errors class can be
    // initialized. When it returns false, that class will fail to initialize due to missing native
    // dependencies.

    val nettyRetryableErrorNumbers = Set(
      // these should match (where an equivalent exists) RETRYABLE_ERRNOS in hailtop/utils/utils.py
      Errors.ERRNO_EPIPE_NEGATIVE,
      Errors.ERRNO_ECONNRESET_NEGATIVE,
      Errors.ERROR_ECONNREFUSED_NEGATIVE,
      Errors.ERROR_ENETUNREACH_NEGATIVE,
    )

    {
      case e: Errors.NativeIoException =>
        // NativeIoException is a subclass of IOException; therefore this case must appear before
        // the IOException case
        //
        // expectedErr appears to be the additive inverse of the errno returned by Linux?
        //
        /* https://github.com/netty/netty/blob/24a0ac36ea91d1aee647d738f879ac873892d829/transport-native-unix-common/src/main/java/io/netty/channel/unix/Errors.java#L49 */
        (nettyRetryableErrorNumbers.contains(e.expectedErr) ||
        /* io.netty.channel.unix.Errors$NativeIoException: readAddress(..) failed: Connection reset
         * by peer */
        e.getMessage.contains("Connection reset by peer"))
      case _: Throwable => false
    }
  } else { case _: Throwable => false }
}
