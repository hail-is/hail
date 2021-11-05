package is.hail.expr.ir.agg

import is.hail.annotations.Region
import is.hail.asm4s.Value
import is.hail.expr.ir.{EmitClassBuilder, EmitCode, EmitCodeBuilder}
import is.hail.io.{BufferSpec, InputBuffer, OutputBuffer, TypedCodecSpec}
import is.hail.types.VirtualTypeWithReq
import is.hail.types.physical.stypes.concrete.SBaseStructPointerValue
import is.hail.types.physical.{PCanonicalArray, PCanonicalNDArray, PCanonicalStruct, PCanonicalTuple, PFloat64, PInt64, PSubsetStruct, PTuple, PType}
import is.hail.types.virtual.{TInt32, Type}

object WhiteningAggState {
  val stateType = PCanonicalStruct(true,
    ("m", PInt64(true)),
    ("w", PInt64(true)),
    ("b", PInt64(true)),
    ("blocksize", PInt64(true)),
    ("curSize", PInt64(true)),
    ("pivot", PInt64(true)),
    ("Q", PCanonicalNDArray(PFloat64(true), 2, true)),
    ("R", PCanonicalNDArray(PFloat64(true), 2, true)),
    ("work1", PCanonicalNDArray(PFloat64(true), 2, true)),
    ("work2", PCanonicalNDArray(PFloat64(true), 2, true)),
    ("Rtemp", PCanonicalNDArray(PFloat64(true), 2, true)),
    ("Qtemp", PCanonicalNDArray(PFloat64(true), 2, true)),
    ("Qtemp2", PCanonicalNDArray(PFloat64(true), 2, true)),
    ("work3", PCanonicalNDArray(PFloat64(true), 1, true)),
    ("T", PCanonicalNDArray(PFloat64(true), 1, true))
  )

  val compactStateType = PCanonicalStruct(true,
    ("m", PInt64(true)),
    ("w", PInt64(true)),
    ("b", PInt64(true)),
    ("blocksize", PInt64(true)),
    ("curSize", PInt64(true)),
    ("pivot", PInt64(true)),
    ("Q", PCanonicalNDArray(PFloat64(true), 2, true)),
    ("R", PCanonicalNDArray(PFloat64(true), 2, true))
  )
}

class WhiteningAggState(val kb: EmitClassBuilder[_]) extends AbstractTypedRegionBackedAggState(WhiteningAggState.stateType) {
  def struct(cb: EmitCodeBuilder): SBaseStructPointerValue = ptype.asInstanceOf[PCanonicalStruct].loadCheapSCode(cb, storageType.loadField(off, 0))
}

//class WhiteningAggregator() extends StagedAggregator {
//  type State = WhiteningAggState
//
//  val matType = PCanonicalNDArray(PFloat64(true), 2, true)
//  val resultType = matType
//  val initOpTypes: Seq[Type] = Array[Type](TInt32, TInt32, TInt32, TInt32)
//  val seqOpTypes: Seq[Type] = Array[Type](matType.virtualType)
//
//  protected def _initOp(cb: EmitCodeBuilder, state: State, args: Array[EmitCode]): Unit = {
//    assert(args.isEmpty)
//    state.bll.init(cb, state.region)
//  }
//
//  protected def _seqOp(cb: EmitCodeBuilder, state: State, seq: Array[EmitCode]): Unit = {
//    state.bll.push(cb, state.region, seq(0))
//  }
//
//  protected def _combOp(cb: EmitCodeBuilder, state: State, other: State): Unit =
//    state.bll.append(cb, state.region, other.bll)
//
//  protected def _storeResult(cb: EmitCodeBuilder, state: State, pt: PType, addr: Value[Long], region: Value[Region], ifMissing: EmitCodeBuilder => Unit): Unit = {
//    assert(pt == resultType)
//    // deepCopy is handled by the blocked linked list
//    pt.storeAtAddress(cb, addr, region, state.bll.resultArray(cb, region, resultType), deepCopy = false)
//  }
//}
