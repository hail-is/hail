package is.hail.lir

import is.hail.HailSuite
import is.hail.expr.ir.{EmitFunctionBuilder, ParamType}
import is.hail.asm4s._
import org.testng.annotations.Test

class LIRSplitSuite extends HailSuite {

  @Test def testSplitPreservesParameterMutation() {
    val f = EmitFunctionBuilder[Unit](ctx, "F")
    f.emitWithBuilder { cb =>
      val mb = f.newEmitMethod("m", IndexedSeq[ParamType](typeInfo[Long]), typeInfo[Unit])
      mb.voidWithBuilder { cb =>
        val arg = mb.getCodeParam[Long](1)

        cb.assign(arg, 1000L)
        (0 until 1000).foreach { i =>
          cb.ifx(arg.cne(1000L), cb._fatal(s"bad split at $i!"))
        }
      }
      cb.invokeVoid(mb, const(1L))
      Code._empty
    }
    f.resultWithIndex()(ctx.fs, 0, ctx.r)()
  }
}
