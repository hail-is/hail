package is.hail.backend

import is.hail.HailContext
import is.hail.annotations.Region
import is.hail.asm4s._

class BackendUtils(mods: Array[(String, (Int, Region) => AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]])]) {

  type F = AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]]

  private[this] val loadedModules: Map[String, (Int, Region) => F] = mods.toMap

  def getModule(id: String): (Int, Region) => F = loadedModules(id)

  def collectDArray(modID: String, contexts: Array[Array[Byte]], globals: Array[Byte]): Array[Array[Byte]] = {
    if (contexts.isEmpty)
      return Array()
    val backend = HailContext.backend
    val globalsBC = backend.broadcast(globals)
    val f = getModule(modID)

    if (contexts.isEmpty) { return Array() }
    backend.parallelizeAndComputeWithIndex(contexts) { (ctx, i) =>
      val gs = globalsBC.value
      Region.scoped { region =>
        val res = f(i, region)(region, ctx, gs)
        res
      }
    }
  }
}
