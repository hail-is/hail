package is.hail.backend

import is.hail.HailContext
import is.hail.annotations.{Region, RegionPool}
import is.hail.asm4s._
import is.hail.expr.ir.lowering.TableStageDependency
import is.hail.io.fs.FS
import is.hail.utils._

object BackendUtils {
  type F = AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]]
}

class BackendUtils(mods: Array[(String, (HailClassLoader, FS, Int, Region) => BackendUtils.F)]) {

  import BackendUtils.F

  private[this] val loadedModules: Map[String, (HailClassLoader, FS, Int, Region) => F] = mods.toMap

  def getModule(id: String): (HailClassLoader, FS, Int, Region) => F = loadedModules(id)

  def collectDArray(backendContext: BackendContext, theDriverHailClassLoader: HailClassLoader, fs: FS, modID: String, contexts: Array[Array[Byte]], globals: Array[Byte], stageName: String, tsd: Option[TableStageDependency]): Array[Array[Byte]] = {
    if (contexts.isEmpty)
      return Array()
    val backend = HailContext.backend
    val f = getModule(modID)

    log.info(s"executing D-Array [$stageName] with ${contexts.length} tasks")
    val t = System.nanoTime()
    val r = if (contexts.length == 0)
      Array.empty[Array[Byte]]
    else if (contexts.length == 1 && backend.canExecuteParallelTasksOnDriver) {
      RegionPool.scoped { rp =>
        rp.scopedRegion { r =>
          Array(f(theDriverHailClassLoader, fs, 0, r)(r, contexts(0), globals))
        }
      }
    } else {
      val globalsBC = backend.broadcast(globals)
      backend.parallelizeAndComputeWithIndex(backendContext, fs, contexts, tsd)({ (ctx, htc, theHailClassLoader, fs) =>
        val gs = globalsBC.value
        htc.getRegionPool().scopedRegion { region =>
          val res = f(theHailClassLoader, fs, htc.partitionId(), region)(region, ctx, gs)
          res
        }
      })
    }

    log.info(s"executed D-Array [$stageName] in ${formatTime(System.nanoTime() - t)}")
    r
  }
}
