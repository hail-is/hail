package is.hail

import java.io._

import scala.util.Random
import is.hail.utils._

package object scheduler {
  def writeByteArray(b: Array[Byte], out: DataOutputStream): Unit = {
    out.writeInt(b.length)
    out.write(b)
  }

  def writeObject[T](v: T, out: DataOutputStream): Unit = {
    val bos = new ByteArrayOutputStream()
    val boos = new ObjectOutputStream(bos)
    boos.writeObject(v)
    writeByteArray(bos.toByteArray, out)
  }

  def readByteArray(in: DataInputStream): Array[Byte] = {
    val n = in.readInt()
    val b = new Array[Byte](n)
    in.readFully(b)
    b
  }

  def readObject[T](in: DataInputStream): T = {
    val b = readByteArray(in)
    val bis = new ByteArrayInputStream(b)
    val bois = new ObjectInputStream(bis)
    bois.readObject().asInstanceOf[T]
  }

  def retry[T](f: () => T, exp: Double = 2.0, maxWait: Double = 60.0): T = {
    var minWait = 1.0
    var w = minWait
    while (true) {
      val startTime = System.nanoTime()
      try {
        return f()
      } catch {
        case b: BreakRetryException =>
          throw b.getCause
        case e: Exception =>
          log.warn(s"retry: restarting due to exception: $e")
          e.printStackTrace()
      }
      val endTime = System.nanoTime()
      val duration = (endTime - startTime) / 1e-9
      w = math.min(maxWait, math.max(minWait, w * exp - duration))
      val t = (1000 * w * Random.nextDouble).toLong
      log.info(s"retry: waiting ${ formatTime(t * 1000000) }")
      Thread.sleep(t)
    }
    null.asInstanceOf[T]
  }

  type Token = Array[Byte]
}
