package is.hail.backend.spark

import is.hail.HailContext
import is.hail.annotations.Region
import is.hail.asm4s._

class SparkBackendUtils(mods: Array[(String, Int => AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]])]) {

  type F = AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]]

  private[this] val loadedModules: Map[String, Int => F] = mods.toMap

  def getModule(id: String): Int => F = loadedModules(id)

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
        val res = f(i)(region, ctx, gs)
        res
      }
    }
  }
}
