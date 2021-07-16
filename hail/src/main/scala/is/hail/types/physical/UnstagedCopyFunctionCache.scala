package is.hail.types.physical

import is.hail.annotations.Region
import is.hail.asm4s.{AsmFunction2RegionLongLong, AsmFunction3RegionLongLongUnit, LongInfo, UnitInfo, classInfo}
import is.hail.expr.ir.{EmitFunctionBuilder, ParamType}
import is.hail.utils.FastIndexedSeq

import scala.collection.mutable

object UnstagedCopyFunctionCache {
  private val compiledCopyFunctions = mutable.Map.empty[(PType, PType, Boolean), AsmFunction2RegionLongLong]
  private val compiledStoreFunctions = mutable.Map.empty[(PType, PType, Boolean), AsmFunction3RegionLongLongUnit]


  def lookupCopy(srcPType: PType, destPType: PType, deepCopy: Boolean): AsmFunction2RegionLongLong = {
    val key = (srcPType, destPType, deepCopy)
    compiledCopyFunctions.get(key) match {
      case Some(f) => f
      case None =>
        this.synchronized {

          if (srcPType.virtualType != destPType.virtualType)
            throw new RuntimeException(s"cannot register a copy function for $srcPType to $this\n  src virt:  ${ srcPType.virtualType }\n  dest virt: ${ destPType.virtualType }")
          val f = EmitFunctionBuilder[AsmFunction2RegionLongLong](null, // ctx can be null if reference genomes and literals do not appear
            "copyFromAddr",
            FastIndexedSeq[ParamType](classInfo[Region], LongInfo), LongInfo)

          f.emitWithBuilder { cb =>
            val region = f.apply_method.getCodeParam[Region](1)
            val srcAddr = f.apply_method.getCodeParam[Long](2)
            destPType.store(cb, region, srcPType.loadCheapSCode(cb, srcAddr), deepCopy = deepCopy)
          }
          val compiledFunction = f.result(allowWorkerCompilation = true)()
          compiledCopyFunctions += ((key, compiledFunction))
          compiledFunction
        }
    }
  }

  def lookupStore(srcPType: PType, destPType: PType, deepCopy: Boolean): AsmFunction3RegionLongLongUnit = {
    val key = (srcPType, destPType, deepCopy)
    compiledStoreFunctions.get(key) match {
      case Some(f) => f
      case None =>
        this.synchronized {
          if (srcPType.virtualType != destPType.virtualType)
            throw new RuntimeException(s"cannot register a copy function for $srcPType to $this\n  src virt:  ${ srcPType.virtualType }\n  dest virt: ${ destPType.virtualType }")
          val f = EmitFunctionBuilder[AsmFunction3RegionLongLongUnit](null, "storeFromAddr", FastIndexedSeq[ParamType](classInfo[Region], LongInfo, LongInfo), UnitInfo)

          f.apply_method.voidWithBuilder { cb =>
            val region = f.apply_method.getCodeParam[Region](1)
            val destAddr = f.apply_method.getCodeParam[Long](2)
            val srcAddr = f.apply_method.getCodeParam[Long](3)
            destPType.storeAtAddress(cb, destAddr, region, srcPType.loadCheapSCode(cb, srcAddr), deepCopy = deepCopy)
          }
          val compiledFunction = f.result(allowWorkerCompilation = true)()
          compiledStoreFunctions += ((key, compiledFunction))
          compiledFunction
        }
    }
  }
}
