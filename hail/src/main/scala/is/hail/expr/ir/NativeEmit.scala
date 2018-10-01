//package is.hail.expr.ir
//
//import is.hail.HailContext
//import is.hail.rvd.RVD
//
//abstract class NativeCode[T]
//abstract class NativeRef[T]
//
//case class NativeEmitPair(missing: NativeCode[Boolean], v: NativeCode[_])
//
//object NativeEmit {
//  type ProcessRowF = NativeRef[Long] => ()
//
//  case class NativeTableEmit(baseRVD: RVD, continuation: ProcessRowF => NativeCode[Unit])
//
//  def emitTable(tir: TableIR, hc: HailContext): NativeTableEmit = tir match {
//    case TableExplode(child: TableIR, fieldName: String) =>
//      val foo = emitTable(child, hc)
//
//      NativeTableEmit(foo.baseRVD, )
//
//
//    case _ =>
//      NativeTableEmit(tir.execute(hc).rvd, asdf)
//
//
//
//
//  }
//
//  def emitValue(ir: IR): NativeEmitPair = ir match {
//    case TableWrite(child, path, overwrite, stageLocally, codecSpecJSONStr) =>
//
//
//
//
//  }
//
//}
