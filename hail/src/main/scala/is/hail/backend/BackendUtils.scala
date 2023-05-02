package is.hail.backend

import is.hail.HailContext
import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.local.LocalTaskContext
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.expr.ir.lowering.TableStageDependency
import is.hail.io.fs._
import is.hail.utils._

import scala.util.Using

object BackendUtils {
  type F = AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]]
}

class BackendUtils(mods: Array[(String, (HailClassLoader, FS, HailTaskContext, Region) => BackendUtils.F)]) {

  import BackendUtils.F

  private[this] val loadedModules: Map[String, (HailClassLoader, FS, HailTaskContext, Region) => F] = mods.toMap

  def getModule(id: String): (HailClassLoader, FS, HailTaskContext, Region) => F = loadedModules(id)

  // add semhash for entire operation
  // consult the cache (semhash -> Array[(int, Array[Byte])]) for those jobs that have completed
  // only pass contexts that have yet to be evaluated
  // update cache with all jobs that passed
  // throw the optional exception
  // return the result
  def collectDArray(backendContext: BackendContext,
                    theDriverHailClassLoader: HailClassLoader,
                    fs: FS,
                    modID: String,
                    _contexts: Array[Array[Byte]],
                    globals: Array[Byte],
                    stageName: String,
                    stageSemanticHash: SemanticHash.Hash.Type,
                    tsd: Option[TableStageDependency]
                   ): Array[Array[Byte]] = {

    val backend = HailContext.backend
    val f = getModule(modID)

    val cachedResults = backendContext.executionCache.lookup(stageSemanticHash)

    val contexts =
      for {
        c@(_, k) <- _contexts.zipWithIndex
        if !cachedResults.containsOrdered[Int](k, _ < _, _._1)
      } yield c

    log.info(s"executing D-Array [$stageName] with ${contexts.length} tasks")
    val t = System.nanoTime()

    val (failure, results): (Option[Throwable], IndexedSeq[(Int, Array[Byte])]) =
      contexts match {
        case Array() =>
          (None, IndexedSeq.empty)

        case Array((context, k)) =>
          Using(new LocalTaskContext(0, 0)) { htc =>
            Using(htc.getRegionPool().getRegion()) { r =>
              FastIndexedSeq((k, f(theDriverHailClassLoader, fs, htc, r)(r, context, globals)))
            }
          }.flatten.fold(t => (Some(t), IndexedSeq.empty), (None, _))

        case _ =>
          val globalsBC = backend.broadcast(globals)
          val fsConfigBC = backend.broadcast(fs.getConfiguration())
          backend.parallelizeAndComputeWithIndex(backendContext, fs, contexts.map(_._1), stageName, tsd) {
            (ctx, htc, theHailClassLoader, fs) =>
              val fsConfig = fsConfigBC.value
              val gs = globalsBC.value
              fs.setConfiguration(fsConfig)
              htc.getRegionPool().scopedRegion { region =>
                f(theHailClassLoader, fs, htc, region)(region, ctx, gs)
              }
          }
      }

    // todo: merge sort these
    val mergedResult = (cachedResults ++ results).sortBy(_._1)
    backendContext.executionCache.put(stageSemanticHash, mergedResult)

    failure.foreach(throw _)

    log.info(s"executed D-Array [$stageName] in ${formatTime(System.nanoTime() - t)}")
    mergedResult.map(_._2).toArray
  }
}
