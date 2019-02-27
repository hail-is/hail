package is.hail.cxx

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import is.hail.HailContext
import is.hail.annotations.Region
import is.hail.expr.ir
import is.hail.nativecode.{NativeModule, NativeStatus, ObjectArray}
import is.hail.utils.using

import scala.collection.mutable

class SparkUtils(mods: Array[(String, NativeModule)]) {

  private[this] val loadedModules: Map[String, NativeModule] = mods.toMap
  def getModule(id: String): (String, Array[Byte]) =
    (loadedModules(id).getKey, loadedModules(id).getBinary)

  def parallelizeComputeCollect(modID: String, bodyf: String, contexts: Array[Array[Byte]], globals: Array[Byte]): Array[Array[Byte]] = {

    val sc = HailContext.get.sc
    val rdd = sc.parallelize[Array[Byte]](contexts, numSlices = contexts.length)
    val (key, bin) = getModule(modID)

    val globalsBC = sc.broadcast(globals)

    rdd.mapPartitionsWithIndex { case (i, ctxIt) =>
      val ctx = ctxIt.next()
      assert(!ctxIt.hasNext)
      val gs = globalsBC.value

      val st = new NativeStatus()
      val mod = new NativeModule(key, bin)
      mod.findOrBuild(st)
      assert(st.ok, st.toString())
      val f = mod.findLongFuncL2(st, bodyf)
      assert(st.ok, st.toString())

      Region.scoped { region =>
        using(new ByteArrayOutputStream()) { baos =>
          val objs = new ObjectArray(baos, new ByteArrayInputStream(ctx), new ByteArrayInputStream(gs))
          f(st, region.get(), objs.get())
          assert(st.ok, st.toString())
          objs.close()
          st.close()
          Iterator.single(baos.toByteArray)
        }
      }
    }.collect()
  }

}
