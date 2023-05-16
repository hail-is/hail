package is.hail.backend

import is.hail.HailContext
import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.backend.local.LocalTaskContext
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.expr.ir.lowering.TableStageDependency
import is.hail.io.fs._
import is.hail.utils._

import scala.util.Try

object BackendUtils {
  type F = AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]]
}

class BackendUtils(mods: Array[(String, (HailClassLoader, FS, HailTaskContext, Region) => BackendUtils.F)]) {

  import BackendUtils.F

  private[this] val loadedModules: Map[String, (HailClassLoader, FS, HailTaskContext, Region) => F] = mods.toMap

  def getModule(id: String): (HailClassLoader, FS, HailTaskContext, Region) => F = loadedModules(id)

  def collectDArray(backendContext: BackendContext,
                    theDriverHailClassLoader: HailClassLoader,
                    fs: FS,
                    modID: String,
                    contexts: Array[Array[Byte]],
                    globals: Array[Byte],
                    stageName: String,
                    semhash: SemanticHash.Type,
                    tsd: Option[TableStageDependency]
                   ): Array[Array[Byte]] = {
    log.info(s"[collectDArray|$stageName]: querying cache for $semhash")
    val cachedResults = backendContext.executionCache.lookup(semhash)
    log.info(s"[collectDArray|$stageName]: found ${cachedResults.length} entries for $semhash.")
    Some {
        for {
          c@(_, k) <- contexts.zipWithIndex
          if !cachedResults.containsOrdered[Int](k, _ < _, _._1)
        } yield c
      }
        .filter(_.nonEmpty)
        .map { remainingContexts =>
          val backend = HailContext.backend
          val f = getModule(modID)

          log.info(
            s"[collectDArray|$stageName]: executing ${remainingContexts.length} , " +
              s"contexts size = ${formatSpace(contexts.map(_.length.toLong).sum)}, " +
              s"globals size = ${formatSpace(globals.length)}"
          )

          val t = System.nanoTime()
          val (failureOpt, successes) =
            remainingContexts match {
              case Array((context, k)) if backend.canExecuteParallelTasksOnDriver =>
                Try {
                  using(new LocalTaskContext(k, 0)) { htc =>
                    using(htc.getRegionPool().getRegion()) { r =>
                      FastIndexedSeq((k, f(theDriverHailClassLoader, fs, htc, r)(r, context, globals)))
                    }
                  }
                }
                  .fold(t => (Some(t), IndexedSeq.empty), (None, _))

              case _ =>
                val globalsBC = backend.broadcast(globals)
                val fsConfigBC = backend.broadcast(fs.getConfiguration())
                val (failureOpt, successes) =
                  backend.parallelizeAndComputeWithIndex(backendContext, fs, remainingContexts.map(_._1), stageName, tsd) {
                    (ctx, htc, theHailClassLoader, fs) =>
                      val fsConfig = fsConfigBC.value
                      val gs = globalsBC.value
                      fs.setConfiguration(fsConfig)
                      htc.getRegionPool().scopedRegion { region =>
                        f(theHailClassLoader, fs, htc, region)(region, ctx, gs)
                      }
                  }
                (failureOpt, successes.map { case (k, v) => (remainingContexts(k)._2, v) })
            }

          log.info(s"[collectDArray|$stageName]: executed ${remainingContexts.length} tasks in ${formatTime(System.nanoTime() - t)}")

          // todo: merge join these
          val results = (cachedResults ++ successes).sortBy(_._1)
          backendContext.executionCache.put(semhash, results)

          failureOpt.foreach(throw _)

          results
        }
      .getOrElse(cachedResults)
      .map(_._2)
      .toArray
  }
}
