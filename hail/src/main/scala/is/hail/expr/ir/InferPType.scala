package is.hail.expr.ir

import is.hail.expr.types.physical._
import is.hail.expr.types.virtual.{TNDArray, TTuple}
import is.hail.utils._

object InferPType {
  def apply(ir: IR, env: Env[PType]): Unit = {
    assert(ir._pType2 == null)
    ir._pType2 = ir match {
      case I32(_) => PInt32(true)
      case I64(_) => PInt64(true)
      case F32(_) => PFloat32(true)
      case F64(_) => PFloat64(true)
      case Str(_) => PString(true)
      case Literal(t, _) => PType.canonical(t, true)
      case True() | False() => PBoolean(true)
      case Void() => PVoid
      case Cast(ir, t) => {
        InferPType(ir, env)
        PType.canonical(t, ir.pType2.required)
      }
      case CastRename(ir, t) => {
        InferPType(ir, env)
        PType.canonical(t, ir.pType2.required)
      }
      case NA(t) => PType.canonical(t, false)
      case IsNA(ir) => {
        InferPType(ir, env)
        PBoolean(true)
      }
      case Ref(name, _) => env.lookup(name)
      case MakeNDArray(data, shape, rowMajor) => {
        InferPType(data, env)
        InferPType(shape, env)
        InferPType(rowMajor, env)

        val nElem = shape.pType2.asInstanceOf[PTuple].size

        PNDArray(coerce[PArray](data.pType2).elementType.setRequired(true), nElem, data.pType2.required && shape.pType2.required)
      }
      case ArrayRange(start: IR, stop: IR, step: IR) => {
        InferPType(start, env)
        InferPType(stop, env)
        InferPType(step, env)

        assert(start.pType2 isOfType stop.pType2)
        assert(start.pType2 isOfType step.pType2)

        val allRequired = start.pType2.required && stop.pType2.required && step.pType2.required
        PArray(start.pType2.setRequired(true), allRequired)
      }
      case StreamRange(start: IR, stop: IR, step: IR) => {
        InferPType(start, env)
        InferPType(stop, env)
        InferPType(step, env)

        assert(start.pType2 isOfType stop.pType2)
        assert(start.pType2 isOfType step.pType2)

        val allRequired = start.pType2.required && stop.pType2.required && step.pType2.required
        PArray(start.pType2.setRequired(true), allRequired)
      }
      case ArrayLen(a: IR) => {
        InferPType(a, env)

        PInt32(a.pType2.required)
      }
      case LowerBoundOnOrderedCollection(orderedCollection: IR, bound: IR, _) => {
        InferPType(orderedCollection, env)
        InferPType(bound, env)

        PInt32(orderedCollection.pType2.required)
      }
      case _: ArrayFor => PVoid
      case _: InitOp => PVoid
      case _: SeqOp => PVoid
      case _: Begin => PVoid
      case Die(_, t) => PType.canonical(t, true)
      case Let(name, value, body) => {
        InferPType(value, env)
        InferPType(body, env.bind(name, value.pType2))

        body.pType2
      }
      case ApplyBinaryPrimOp(op, l, r) => {
          InferPType(l, env)
          InferPType(r, env)

          val required = l.pType2.required && r.pType2.required
          val vType = BinaryOp.getReturnType(op, l.pType2.virtualType, r.pType2.virtualType).setRequired(required)

          PType.canonical(vType, vType.required)
      }
      case ApplyUnaryPrimOp(op, v) => {
        InferPType(v, env)
        PType.canonical(UnaryOp.getReturnType(op, v.pType2.virtualType).setRequired(v.pType2.required))
      }
      case ApplyComparisonOp(op, l, r) => {
        InferPType(l, env)
        InferPType(r, env)

        assert(l.pType2 isOfType r.pType2)
        op match {
          case _: Compare => PInt32(l.pType2.required && r.pType2.required)
          case _ => PBoolean(l.pType2.required && r.pType2.required)
        }
      }
      case a: ApplyIR => {
        InferPType(a.explicitNode, env)
        a.explicitNode.pType2
      }
      case a: AbstractApplyNode[_] => {
        val pTypes = a.args.map( i => {
          InferPType(i, env)
          i.pType2
        })
        a.implementation.returnPType(pTypes)
      }
      case a@ApplySpecial(_, args, _) => {
        val pTypes = args.map( i => {
          InferPType(i, env)
          i.pType2
        })
        a.implementation.returnPType(pTypes)
      }
      case _: Uniroot => PFloat64()
      case ArrayRef(a, i) => {
        InferPType(a, env)
        InferPType(i, env)
        assert(i.pType2 isOfType PInt32() )

        coerce[PStreamable](a.pType2).elementType.setRequired(a.pType2.required && i.pType2.required)
      }
      case ArraySort(a, leftName, rightName, compare) => {
        InferPType(a, env)
        val et = coerce[PStreamable](a.pType2).elementType

        InferPType(compare, env.bind(leftName -> et, rightName -> et))
        assert(compare.pType2.isOfType(PBoolean()))

        PArray(et, a.pType2.required)
      }
      case ToSet(a) => {
        InferPType(a, env)
        val et = coerce[PIterable](a.pType2).elementType
        PSet(et, a.pType2.required)
      }
      case ToDict(a) => {
        InferPType(a, env)
        val elt = coerce[PBaseStruct](coerce[PIterable](a.pType2).elementType)
        PDict(elt.types(0), elt.types(1), a.pType2.required)
      }
      case ToArray(a) => {
        InferPType(a, env)
        val elt = coerce[PIterable](a.pType2).elementType
        PArray(elt, a.pType2.required)
      }
      case ToStream(a) => {
        InferPType(a, env)
        val elt = coerce[PIterable](a.pType2).elementType
        PStream(elt, a.pType2.required)
      }
      case GroupByKey(collection) => {
        InferPType(collection, env)
        val elt = coerce[PBaseStruct](coerce[PStreamable](collection.pType2).elementType)
        PDict(elt.types(0), PArray(elt.types(1)), collection.pType2.required)
      }
      case ArrayMap(a, name, body) => {
        InferPType(a, env)
        InferPType(body, env.bind(name, a.pType2.asInstanceOf[PArray].elementType))
        coerce[PStreamable](a.pType2).copyStreamable(body.pType2, body.pType2.required)
      }
      case ArrayFilter(a, name, cond) => {
        InferPType(a, env)
        a.pType2
      }
      case ArrayFlatMap(a, name, body) => {
        InferPType(a, env)
        InferPType(body, env.bind(name, a.pType2.asInstanceOf[PArray].elementType))

        // Whether an array must return depends on a, but element requiredeness depends on body (null a elements elided)
        coerce[PStreamable](a.pType2).copyStreamable(coerce[PIterable](body.pType2).elementType, a.pType2.required)
      }
      case ArrayFold(a, zero, accumName, valueName, body) => {
        InferPType(zero, env)

        InferPType(a, env)
        InferPType(body, env.bind(accumName -> zero.pType2, valueName -> a.pType2.asInstanceOf[PArray].elementType))
        assert(body.pType2 isOfType zero.pType2)

        zero.pType2.setRequired(body.pType2.required)
      }
      case ArrayScan(a, zero, accumName, valueName, body) => {
        InferPType(zero, env)

        InferPType(a, env)
        InferPType(body, env.bind(accumName -> zero.pType2, valueName -> a.pType2.asInstanceOf[PArray].elementType))
        assert(body.pType2 isOfType zero.pType2)

        val elementPType = zero.pType2.setRequired(body.pType2.required && zero.pType2.required)
        coerce[PStreamable](a.pType2).copyStreamable(elementPType, a.pType2.required)
      }
      case ArrayLeftJoinDistinct(lIR, rIR, lName, rName, compare, join) => {
        InferPType(lIR, env)
        InferPType(rIR, env)

        InferPType(join, env.bind(lName -> lIR.pType2.asInstanceOf[PArray].elementType, rName -> rIR.pType2.asInstanceOf[PArray].elementType))

        PArray(join.pType2, lIR.pType2.required)
      }
      case NDArrayShape(nd) => {
        InferPType(nd, env)
        PTuple(nd.pType2.required, IndexedSeq.tabulate(nd.pType2.asInstanceOf[PNDArray].nDims)(_ => PInt64(true)):_*)
      }
      case NDArrayReshape(nd, shape) => {
        InferPType(nd, env)
        InferPType(shape, env)

        PNDArray(coerce[PNDArray](nd.pType2).elementType, shape.pType2.asInstanceOf[TTuple].size, nd.pType2.required)
      }
      case NDArrayMap(nd, name, body) => {
        InferPType(nd, env)
        InferPType(body, env.bind(name, nd.pType2))

        PNDArray(body.pType2, coerce[PNDArray](nd.pType2).nDims, nd.pType2.required)
      }
      case NDArrayMap2(l, r, lName, rName, body) => {
        InferPType(l, env)
        InferPType(body, env)

        PNDArray(body.pType2, coerce[PNDArray](l.pType2).nDims, l.pType2.required)
      }
      case NDArrayReindex(nd, indexExpr) => {
        InferPType(nd, env)

        PNDArray(coerce[PNDArray](nd.pType2).elementType, indexExpr.length, nd.pType2.required)
      }
      case NDArrayRef(nd, idxs) => {
        InferPType(nd, env)
        
        var allRequired = nd.pType2.required
        val it = idxs.iterator
        while(it.hasNext) {
          val idxIR = it.next()

          InferPType(idxIR, env)

          assert(idxIR.pType2.isOfType(PInt64()) || idxIR.pType2.isOfType(PInt32()))
          
          if(allRequired == true && idxIR.pType2.required == false) {
            allRequired = false
          }
        }

        coerce[PNDArray](nd.pType2).elementType.setRequired(allRequired)
      }
      case NDArraySlice(nd, slices) => {
        InferPType(nd, env)
        InferPType(slices, env)

        val remainingDims = coerce[PTuple](slices.pType2).types.filter(_.isInstanceOf[PTuple])

        PNDArray(coerce[PNDArray](nd.pType2).elementType, remainingDims.length, remainingDims.forall(_.required))
      }
      case NDArrayMatMul(l, r) => {
        InferPType(l, env)
        InferPType(r, env)
        val lTyp = coerce[PNDArray](l.pType2)
        val rTyp = coerce[PNDArray](r.pType2)
        PNDArray(lTyp.elementType, TNDArray.matMulNDims(lTyp.nDims, rTyp.nDims), lTyp.required && rTyp.required)
      }
      case NDArrayWrite(_, _) => PVoid
      case MakeStruct(fields) => PStruct(true, fields.map {
        case (name, a) => {
          InferPType(a, env)

          (name, a.pType2)
        }
      }: _*)
      case SelectFields(old, fields) => {
        InferPType(old, env)
        val tbs = coerce[PStruct](old.pType2)
        tbs.select(fields.toFastIndexedSeq)._1
      }
      case InsertFields(old, fields, fieldOrder) => {
        InferPType(old, env)
        val tbs = coerce[PStruct](old.pType2)

        val s = tbs.insertFields(fields.map(f =>  {
          InferPType(f._2, env)
          (f._1, f._2.pType2)
        }))

        fieldOrder.map { fds =>
          assert(fds.length == s.size)
          PStruct(fds.map(f => f -> s.fieldType(f)): _*)
        }.getOrElse(s)
      }
      case GetField(o, name) => {
        InferPType(o, env)
        val t = coerce[PStruct](o.pType2)
        if (t.index(name).isEmpty)
          throw new RuntimeException(s"$name not in $t")
        val fd = t.field(name).typ
        fd.setRequired(t.required && fd.required)
      }
      case MakeTuple(values) => PTuple(true, values.map(v => {
          InferPType(v._2, env)
          v._2.pType2
      }):_*)
      case GetTupleElement(o, idx) => {
        InferPType(o, env)
        val t = coerce[PTuple](o.pType2)
        assert(idx >= 0 && idx < t.size)
        val fd = t.types(idx)
        fd.setRequired(t.required && fd.required)
      }
      case CollectDistributedArray(contexts, globals, contextsName, globalsName, body) => {
        InferPType(contexts, env)
        InferPType(globals, env)

        InferPType(body, env.bind(contextsName -> contexts.pType2, globalsName -> globals.pType2))
        PArray(body.pType2)
      }
      case ReadPartition(_, _, rowType) => throw new Exception("node not supported")
      case _: Coalesce | _: MakeArray | _: MakeStream | _: If => throw new Exception("Node not supported")
    }

    // Allow only requiredeness to diverge
    assert(ir.pType2.virtualType isOfType ir.typ)
  }
}

