package is.hail.backend

import is.hail.annotations.Region
import is.hail.asm4s._
import is.hail.expr.ir.analyses.SemanticHash
import is.hail.expr.ir.lowering.TableStageDependency
import is.hail.io.fs._
import is.hail.utils._

object BackendUtils {
  type F = AsmFunction3[Region, Array[Byte], Array[Byte], Array[Byte]]
}

class BackendUtils(
  mods: Array[(String, (HailClassLoader, FS, HailTaskContext, Region) => BackendUtils.F)]
) {

  import BackendUtils.F

  private[this] val loadedModules
    : Map[String, (HailClassLoader, FS, HailTaskContext, Region) => F] = mods.toMap

  def getModule(id: String): (HailClassLoader, FS, HailTaskContext, Region) => F = loadedModules(id)

  def collectDArray(
    ctx: DriverContext,
    modID: String,
    contexts: Array[Array[Byte]],
    globals: Array[Byte],
    stageName: String,
    semhash: Option[SemanticHash.Type],
    tsd: Option[TableStageDependency],
  ): Array[Array[Byte]] = {

    val cachedResults =
      semhash
        .map { s =>
          log.info(s"[collectDArray|$stageName]: querying cache for $s")
          val cachedResults = ctx.executionCache.lookup(s)
          log.info(s"[collectDArray|$stageName]: found ${cachedResults.length} entries for $s.")
          cachedResults
        }
        .getOrElse(IndexedSeq.empty)

    val remainingPartitions =
      contexts.indices.filterNot(k => cachedResults.containsOrdered[Int](k, _ < _, _._2))

    val mod = getModule(modID)
    val t = System.nanoTime()
    val (failureOpt, successes) =
      remainingPartitions match {
        case Seq() =>
          (None, IndexedSeq.empty)

        case partitions =>
          ctx.collectDistributedArray(
            globals,
            contexts,
            stageName,
            tsd,
            Some(partitions),
          ) { (gs, ctx, htc, theHailClassLoader, fs) =>
            htc.getRegionPool().scopedRegion { region =>
              mod(theHailClassLoader, fs, htc, region)(region, ctx, gs)
            }
          }
      }

    log.info(
      s"[collectDArray|$stageName]: executed ${remainingPartitions.length} tasks in ${formatTime(System.nanoTime() - t)}"
    )

    val results =
      merge[(Array[Byte], Int)](
        cachedResults,
        successes.sortBy(_._2),
        _._2 < _._2,
      )

    semhash.foreach(s => ctx.executionCache.put(s, results))
    failureOpt.foreach(throw _)

    results.map(_._1).toArray
  }
}
