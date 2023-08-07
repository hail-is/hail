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
                    semhash: Option[SemanticHash.Type],
                    tsd: Option[TableStageDependency]
                   ): Array[Array[Byte]] = {

    val cachedResults =
      semhash
        .map { s =>
          log.info(s"[collectDArray|$stageName]: querying cache for $s")
          val cachedResults = backendContext.executionCache.lookup(s)
          log.info(s"[collectDArray|$stageName]: found ${cachedResults.length} entries for $s.")
          cachedResults
        }
        .getOrElse(IndexedSeq.empty)

    val remainingContexts =
      for {
        c@(_, k) <- contexts.zipWithIndex
        if !cachedResults.containsOrdered[Int](k, _ < _, _._2)
      } yield c

    val results =
      if (remainingContexts.isEmpty) cachedResults else {
        val backend = HailContext.backend
        val f = getModule(modID)

        log.info(
          s"[collectDArray|$stageName]: executing ${remainingContexts.length} tasks, " +
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
                    val run = f(theDriverHailClassLoader, fs, htc, r)
                    val res = is.hail.services.retryTransientErrors {
                      run(r, context, globals)
                    }
                    FastSeq(res -> k)
                  }
                }
              }
                .fold(t => (Some(t), IndexedSeq.empty), (None, _))

            case _ =>
              val globalsBC = backend.broadcast(globals)
              val fsConfigBC = backend.broadcast(fs.getConfiguration())
              val (failureOpt, successes) =
                backend.parallelizeAndComputeWithIndex(backendContext, fs, remainingContexts, stageName, tsd) {
                  (ctx, htc, theHailClassLoader, fs) =>
                    val fsConfig = fsConfigBC.value
                    val gs = globalsBC.value
                    fs.setConfiguration(fsConfig)
                    htc.getRegionPool().scopedRegion { region =>
                      f(theHailClassLoader, fs, htc, region)(region, ctx, gs)
                    }
                }
              (failureOpt, successes)
          }

        log.info(s"[collectDArray|$stageName]: executed ${remainingContexts.length} tasks " +
          s"in ${formatTime(System.nanoTime() - t)}"
        )

        val results = merge[(Array[Byte], Int)](cachedResults, successes.sortBy(_._2), _._2 < _._2)
        semhash.foreach(s => backendContext.executionCache.put(s, results))
        failureOpt.foreach(throw _)

        results
      }

      results.map(_._1).toArray
  }
}