package is.hail.cxx

import java.io.{ByteArrayInputStream, ByteArrayOutputStream}

import is.hail.HailContext
import is.hail.annotations.Region
import is.hail.nativecode.{NativeModule, NativeStatus, ObjectArray}
import is.hail.utils.using

class SparkUtils(mods: Array[(String, (Array[Byte], NativeModule))]) {

  type Literals = Array[Byte]

  private[this] val loadedModules: Map[String, (Literals, NativeModule)] = mods.toMap
  def getModule(id: String): (Literals, String, Array[Byte]) = {
    val (lit, mod) = loadedModules(id)
    (lit, mod.getKey, mod.getBinary)
  }

  def parallelizeComputeCollect(modID: String, bodyf: String, contexts: Array[Array[Byte]], globals: Array[Byte]): Array[Array[Byte]] = {


    val backend = HailContext.backend
    val globalsBC = backend.broadcast(globals)
    val (lit, key, bin) = getModule(modID)

    backend.parallelizeAndComputeWithIndex(contexts) { (ctx, i) =>
      val gs = globalsBC.value

      val st = new NativeStatus()
      val mod = new NativeModule(key, bin)
      mod.findOrBuild(st)
      assert(st.ok, st.toString())
      val f = mod.findLongFuncL2(st, bodyf)
      assert(st.ok, st.toString())

      Region.scoped { region =>
        using(new ByteArrayOutputStream()) { baos =>
          val objs = new ObjectArray(baos, new ByteArrayInputStream(ctx), new ByteArrayInputStream(gs), new ByteArrayInputStream(lit))
          f(st, region.get(), objs.get())
          assert(st.ok, st.toString())
          objs.close()
          st.close()
          baos.toByteArray
        }
      }
    }
  }

}
