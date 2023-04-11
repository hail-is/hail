package is.hail.services

import io.netty.channel.unix.Errors  // cannot be in package.scala because is.hail.io shadows top-level io
import io.netty.channel.unix.Errors.NativeIoException  // cannot be in package.scala because is.hail.io shadows top-level io

object NettyProxy {
  type Errors = io.netty.channel.unix.Errors
  type NativeIoException = io.netty.channel.unix.Errors.NativeIoException
}
