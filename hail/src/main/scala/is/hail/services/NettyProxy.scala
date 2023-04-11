package is.hail.services

import io.netty.channel.unix.Errors  // cannot be in package.scala because is.hail.io shadows top-level io
import io.netty.channel.unix.Errors.NativeIoException  // cannot be in package.scala because is.hail.io shadows top-level io

object NettyProxy {
  val ERRNO_ENOENT_NEGATIVE = io.netty.channel.unix.Errors.ERRNO_ENOENT_NEGATIVE
  val ERRNO_ENOTCONN_NEGATIVE = io.netty.channel.unix.Errors.ERRNO_ENOTCONN_NEGATIVE
  val ERRNO_EBADF_NEGATIVE = io.netty.channel.unix.Errors.ERRNO_EBADF_NEGATIVE
  val ERRNO_EPIPE_NEGATIVE = io.netty.channel.unix.Errors.ERRNO_EPIPE_NEGATIVE
  val ERRNO_ECONNRESET_NEGATIVE = io.netty.channel.unix.Errors.ERRNO_ECONNRESET_NEGATIVE
  val ERRNO_EAGAIN_NEGATIVE = io.netty.channel.unix.Errors.ERRNO_EAGAIN_NEGATIVE
  val ERRNO_EWOULDBLOCK_NEGATIVE = io.netty.channel.unix.Errors.ERRNO_EWOULDBLOCK_NEGATIVE
  val ERRNO_EINPROGRESS_NEGATIVE = io.netty.channel.unix.Errors.ERRNO_EINPROGRESS_NEGATIVE
  val ERROR_ECONNREFUSED_NEGATIVE = io.netty.channel.unix.Errors.ERROR_ECONNREFUSED_NEGATIVE
  val ERROR_EISCONN_NEGATIVE = io.netty.channel.unix.Errors.ERROR_EISCONN_NEGATIVE
  val ERROR_EALREADY_NEGATIVE = io.netty.channel.unix.Errors.ERROR_EALREADY_NEGATIVE
  val ERROR_ENETUNREACH_NEGATIVE = io.netty.channel.unix.Errors.ERROR_ENETUNREACH_NEGATIVE

  type NativeIoException = io.netty.channel.unix.Errors.NativeIoException
}
