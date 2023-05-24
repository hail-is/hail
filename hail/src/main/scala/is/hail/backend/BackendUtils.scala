package is.hail.backend

import is.hail.HailContext
import is.hail.annotations.{Region, RegionPool}
import is.hail.asm4s._
import is.hail.backend.local.LocalTaskContext
import is.hail.expr.ir.lowering.TableStageDependency
import is.hail.io.fs._
import is.hail.utils._

import scala.reflect.ClassTag

object BackendUtils {
  type F = AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]]
}

class BackendUtils(mods: Array[(String, (HailClassLoader, FS, HailTaskContext, Region) => BackendUtils.F)]) {

  import BackendUtils.F

  private[this] val loadedModules: Map[String, (HailClassLoader, FS, HailTaskContext, Region) => F] = mods.toMap

  def getModule(id: String): (HailClassLoader, FS, HailTaskContext, Region) => F = loadedModules(id)

  def collectDArray(backendContext: BackendContext, theDriverHailClassLoader: HailClassLoader, fs: FS, modID: String, contexts: Array[Array[Byte]], globals: Array[Byte], stageName: String, tsd: Option[TableStageDependency]): Array[Array[Byte]] = {
    if (contexts.isEmpty)
      return Array()
    val backend = HailContext.backend
    val f = getModule(modID)

    log.info(s"executing D-Array [$stageName] with ${contexts.length} tasks, " +
      s"contexts size = ${formatSpace(contexts.map(_.length.toLong).sum)}, globals size = ${formatSpace(globals.length)}")
    val t = System.nanoTime()
    val r = if (contexts.length == 0)
      Array.empty[Array[Byte]]
    else if (contexts.length == 1 && backend.canExecuteParallelTasksOnDriver) {
      using(new LocalTaskContext(0, 0)) { htc =>
        using(htc.getRegionPool().getRegion()) { r =>
          Array(f(theDriverHailClassLoader, fs, htc, r)(r, contexts(0), globals))
        }
      }
    } else {
      val globalsBC = backend.broadcast(globals)
      val fsConfigBC = backend.broadcast(fs.getConfiguration())
      backend.parallelizeAndComputeWithIndex(backendContext, fs, contexts, stageName, tsd)({ (ctx, htc, theHailClassLoader, fs) =>
        val fsConfig = fsConfigBC.value
        val gs = globalsBC.value
        fs.setConfiguration(fsConfig)
        htc.getRegionPool().scopedRegion { region =>
          val res = f(theHailClassLoader, fs, htc, region)(region, ctx, gs)
          res
        }
      })
    }

    log.info(s"executed D-Array [$stageName] in ${formatTime(System.nanoTime() - t)}")
    r
  }
}
