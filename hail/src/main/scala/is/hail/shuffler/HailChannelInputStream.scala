package is.hail.shuffler

import java.nio.channels.ReadableByteChannel
import sun.nio.ch.ChannelInputStream

class HailChannelInputStream (
  val channel: ReadableByteChannel
) extends ChannelInputStream(channel) { }
