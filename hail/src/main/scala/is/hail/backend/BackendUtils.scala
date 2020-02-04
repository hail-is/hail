package is.hail.backend

import java.io.{ByteArrayInputStream, ByteArrayOutputStream, InputStream, OutputStream}

import is.hail.HailContext
import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.io.{Decoder, DecoderBuilder}

import scala.collection.mutable

class BroadcastHailValue(val bc: BroadcastValue[(InputStream => Decoder, Array[Byte])]) extends AnyVal {
  def toRegion(r: Region): Long = {
    val (makeDec, value) = bc.value
    makeDec(new ByteArrayInputStream(value)).readRegionValue(r)
  }
}

class BackendUtils(mods: Array[(String, (Int, Region) => AsmFunction4[Region, Long, InputStream, OutputStream, Unit])]) {
  type F = AsmFunction4[Region, Long, InputStream, OutputStream, Unit]

  private[this] val loadedModules: Map[String, (Int, Region) => F] = mods.toMap
  private[this] val broadcastValues: mutable.Map[String, BroadcastHailValue] = new mutable.HashMap()

  def getModule(id: String): (Int, Region) => F = loadedModules(id)

  def addBroadcast(id: String, makeDec: DecoderBuilder, encoded: Array[Byte]): Unit =
    broadcastValues += id -> new BroadcastHailValue(HailContext.backend.broadcast(makeDec.v -> encoded))

  def getBroadcast(id: String): Region => Long = {
    val bc = broadcastValues(id)
    bc.toRegion
  }

  def removeBroadcast(id: String): Unit =
    broadcastValues.remove(id)

  def collectDArray(modID: String, contexts: Array[Array[Byte]], globals: String): Array[Array[Byte]] = {
    if (contexts.isEmpty)
      return Array()
    val backend = HailContext.backend
    val f = getModule(modID)
    val gsBC = getBroadcast(globals)

    if (contexts.isEmpty) { return Array() }
    backend.parallelizeAndComputeWithIndex(contexts) { (ctx, i) =>
      Region.scoped { region =>
        val gs = gsBC(region)
        val ctxIS = new ByteArrayInputStream(ctx)
        val os = new ByteArrayOutputStream()
        f(i, region)(region, gs, ctxIS, os)
        os.toByteArray
      }
    }
  }
}
