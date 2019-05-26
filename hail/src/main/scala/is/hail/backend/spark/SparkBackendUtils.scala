package is.hail.backend.spark

import is.hail.HailContext
import is.hail.annotations.Region
import is.hail.asm4s._

class SparkBackendUtils(mods: Array[(String, Int => AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]])]) {

  type F = AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]]

  private[this] val loadedModules: Map[String, Int => F] = mods.toMap

  def getModule(id: String): Int => F = loadedModules(id)

  def collectDArray(modID: String, contexts: Array[Array[Byte]], globals: Array[Byte]): Array[Array[Byte]] = {
    val sc = HailContext.get.sc
    val rdd = sc.parallelize[Array[Byte]](contexts, numSlices = contexts.length)
    val f = getModule(modID)

    val globalsBC = sc.broadcast(globals)

    rdd.mapPartitionsWithIndex { case (i, ctxIt) =>
      val ctx = ctxIt.next()
      assert(!ctxIt.hasNext)
      val gs = globalsBC.value

      Region.scoped { region =>
        val res = f(i)(region, ctx, gs)
        Iterator.single(res)
      }
    }.collect()
  }
}
