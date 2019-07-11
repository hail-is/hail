package is.hail.expr.ir

import is.hail.expr.types.physical._
import is.hail.expr.types.virtual.{TArray, TTuple}
import is.hail.utils._

// Env contains all of the symbols values to reference
// at root, the env is empty
// In body of say tablemaprows, you are allowed to reference row, or global, etc
// the names of row and global are bound in the body of tablemaprows
// so tablemaprows appends to the environment
// It's possible for a name to be bound twice
// we may have nested array maps, and say each binds i
// inside the inner map body, any reference to the name
// FIXME: strip all requiredness logic when possible
// TODO: We don't implicitly walk bottom-up for virtualTypes (not calling .typ) in all cases
// TODO: If virtual type inference doesn't occur before this step, there may be subtle
// TODO: behavior differences between IR Nodes (those that need .typ here and those that don't)
// TODO: In cases where env is sent as is to child nodes, do we need copy? I believe no, as long as copy-on-write

// TODO: aggregators: 1) use bindAgg/bindScan instead of bindEval? 2) proper passing of env, seems little needed
object InferPType {
  def apply(ir: IR, env: BindingEnv[PType]): PType = {
    ir match {
      case I32(_) => PInt32()
      case I64(_) => PInt64()
      case F32(_) => PFloat32()
      case F64(_) => PFloat64()
      case Str(_) => PString()
      // TODO: The following (where canonical) have no children
      // TODO: OK to be canonical for now? Later we will want to
      // TODO: use properties of env?
      case Literal(t, _) => PType.canonical(t)
      case True() | False() => PBoolean()
      case Void() => PVoid
      case Cast(_, t) => PType.canonical(t)
      case CastRename(_, t) => PType.canonical(t)
      case NA(t) => PType.canonical(t)
      case IsNA(_) => PBoolean()
      case Coalesce(values) => {
        val vit = values.iterator
        val head = values.iterator.next()
        head.inferSetPType(env)

        while(vit.hasNext) {
          val value = vit.next()
          value.inferSetPType(env)

          assert(head.pType == value.pType)
        }
      }
      // TODO: Why is there a type (var _type) on Ref? Cache to avoid lookup? if so seems area for bugs
      case Ref(name, _) => {
        env.eval.lookup(name)
      }
      case In(_, t) => PType.canonical(t)
      // TODO: need to descend into args?
      case MakeArray(_, t) => PType.canonical(t)
      case MakeStream(_, t) => PType.canonical(t)
      case MakeNDArray(data, shape, _) => {
        data.inferSetPType(env)
        shape.inferSetPType(env)

        // TODO: Is this wrong?
        val nElem = shape.pType.asInstanceOf[PTuple].size

        // TODO: requiredeness?
        PNDArray(coerce[PArray](data.pType).elementType, nElem)
      }

      case _: ArrayLen => PInt32()
      case _: ArrayRange => PArray(PInt32())
      case _: StreamRange => PStream(PInt32())
      case _: LowerBoundOnOrderedCollection => PInt32()
      case _: ArrayFor => PVoid
      case _: InitOp => PVoid
      case _: SeqOp => PVoid
      case _: Begin => PVoid
      // FIXME in IR suggests Die should give type any, fix this once that is fixed
      // but in test, TFloat64() is used, which will fit
      case Die(_, t) => PType.canonical(t)
      case If(cond, cnsq, altr) => {
        cond.inferSetPType(env)
        cnsq.inferSetPType(env)
        altr.inferSetPType(env)

        assert(cond.pType == PBoolean())

        if (cnsq.pType != altr.pType)
          cnsq.pType.deepOptional()
        else {
          assert(cnsq.pType isOfType altr.pType)
          cnsq.pType
        }
      }
      case Let(name, value, body) => {
        value.inferSetPType(env)
        body.inferSetPType(env.bindEval(name, value.pType))
        body.pType
      }
      case AggLet(name, value, body, _) => {
        value.inferSetPType(env)
        body.inferSetPType(env.bindEval(name, value.pType))
        body.pType
      }
      case ApplyBinaryPrimOp(op, l, r) => {
          l.inferSetPType(env)
          r.inferSetPType(env)

          val required = l.pType.required && r.pType.required
          val vType = BinaryOp.getReturnType(op, l.pType.virtualType, r.pType.virtualType).setRequired(required)

          PType.canonical(vType)
      }
      case ApplyUnaryPrimOp(op, v) => {
        v.inferSetPType(env)
        PType.canonical(UnaryOp.getReturnType(op, v.pType.virtualType).setRequired(v.pType.required))
      }
      case ApplyComparisonOp(op, l, r) => {
        l.inferSetPType(env)
        r.inferSetPType(env)
        // TODO: should this be l.pType == r.pType or isOfType
        assert(l.pType isOfType r.pType)
        op match {
          case _: Compare => PInt32(l.pType.required && r.pType.required)
          case _ => PBoolean(l.pType.required && r.pType.required)
        }
      }

      case a: ApplyIR => {
        a.explicitNode.inferSetPType(env)
        a.pType
      }
      // TODO: this seems wrong:
      // 1) it may have subtle bugs, .implementation calls ir.typ, which is not pre-filled
      // 2) .subst sets requiredness on virtualTypes
      // 3) pType inference isn't really used to any effect here
      case a: AbstractApplyNode[_] => {
        val argPTypes = a.args.map( ir => {
          ir.inferSetPType(env)
          ir.pType.virtualType
        })
        a.implementation.unify(argPTypes)
        // TODO: This calls setRequired on the virtual types
        PType.canonical(a.implementation.returnType.subst())
      }
      case _: Uniroot => PFloat64()
      // TODO: check that i.pType isOfType is correct (for ArrayRef, ArraySort)
      case ArrayRef(a, i) => {
        a.inferSetPType(env)
        i.inferSetPType(env)
        assert(i.pType isOfType PInt32() )
        coerce[PStreamable](a.pType).elementType.setRequired(a.pType.required && i.pType.required)
      }
      case ArraySort(a, _, _, compare) => {
        a.inferSetPType(env)
        compare.inferSetPType(env)
        assert(compare.pType.isOfType(PBoolean()))
        val et = coerce[PStreamable](a.pType).elementType
        PArray(et, a.pType.required)
      }
      case ToSet(a) => {
        a.inferSetPType(env)
        val et = coerce[PIterable](a.pType).elementType
        PSet(et, a.pType.required)
      }
      case ToDict(a) => {
        a.inferSetPType(env)
        val elt = coerce[PBaseStruct](coerce[PIterable](a.pType).elementType)
        PDict(elt.types(0), elt.types(1), a.pType.required)
      }
      case ToArray(a) => {
        a.inferSetPType(env)
        val elt = coerce[PIterable](a.pType).elementType
        PArray(elt, a.pType.required)
      }
      case ToStream(a) => {
        a.inferSetPType(env)
        val elt = coerce[PIterable](a.pType).elementType
        PStream(elt, a.pType.required)
      }
      case GroupByKey(collection) => {
        collection.inferSetPType(env)
        val elt = coerce[PBaseStruct](coerce[PStreamable](collection.pType).elementType)
        PDict(elt.types(0), PArray(elt.types(1)), collection.pType.required)
      }
      // TODO: Check the env.bindEval needs name, a
      case ArrayMap(a, name, body) => {
        // infer array side of tree fully
        a.inferSetPType(env)
        // push do the element side, applies to each element
        body.inferSetPType(env.bindEval(name, a.pType))
        // TODO: why setRequired false?
        coerce[PStreamable](a.pType).copyStreamable(body.pType.setRequired(false))
      }

      case ArrayFilter(a, name, cond) => {
        a.inferSetPType(env)
        a.pType
      }
      // TODO: Check the env.bindEval needs name, a
      case ArrayFlatMap(a, name, body) => {
        a.inferSetPType(env)
        body.inferSetPType(env.bindEval(name, a.pType))
        coerce[PStreamable](a.pType).copyStreamable(coerce[PIterable](body.pType).elementType)
      }
      case ArrayFold(a, zero, accumName, valueName, body) => {
        a.inferSetPType(env)
        zero.inferSetPType(env)

        body.inferSetPType(env.bindEval(accumName -> zero.pType, valueName -> a.pType))

        assert(body.pType == zero.pType)
        zero.pType
      }
      case ArrayScan(a, zero, accumName, valueName, body) => {
        a.inferSetPType(env)
        zero.inferSetPType(env)

        body.inferSetPType(env.bindEval(accumName -> zero.pType, valueName -> a.pType))

        assert(body.pType == zero.pType)

        coerce[PStreamable](a.pType).copyStreamable(zero.pType)
      }
      case ArrayAgg(a, name, query) => {
        a.inferSetPType(env)
        // TODO: figure out use of bindAgg vs bindEval
        // In PruneDeadFields, sometimes scan and agg are flatteneed into eval, ex:
        //        unifyEnvs(
        //          BindingEnv(eval = concatEnvs(Array(queryEnv.eval.delete(name), queryEnv.scanOrEmpty.delete(name)))),
        //          aEnv)
        // That seems reasonable, to allow us to look up env in only .eval
        query.inferSetPType(env.bindAgg(name, a.pType))
        query.pType
      }
      case ArrayAggScan(a, name, query) => {
        a.inferSetPType(env)
        // TODO: figure out use of bindScan vs bindEval
        query.inferSetPType(env.bindScan(name, a.pType))
        PArray(query.pType)
      }
      case ArrayLeftJoinDistinct(lIR, rIR, lName, rName, compare, join) => {
        lIR.inferSetPType(env)
        rIR.inferSetPType(env)

        assert(lIR.pType == rIR.pType)

        // TODO: Does left / right need same type? seems yes
        // TODO: if so, join can depend on only on left
        join.inferSetPType(env.bindEval(lName, lIR.pType))


        // TODO: Not sure why InferType used copyStreamable here
        PArray(join.pType)
      }
      case NDArrayShape(nd) => {
        nd.inferSetPType(env)
        PTuple(IndexedSeq.tabulate(nd.pType.asInstanceOf[PNDArray].nDims)(_ => PInt64()), nd.pType.required)
      }
      case NDArrayReshape(nd, shape) => {
        nd.inferSetPType(env)
        shape.inferSetPType(env)

        PNDArray(coerce[PNDArray](nd.pType).elementType, shape.pType.asInstanceOf[TTuple].size, nd.pType.required)
      }
      case NDArrayMap(nd, name, body) => {
        nd.inferSetPType(env)
        body.inferSetPType(env.bindEval(name, nd.pType))

        PNDArray(body.pType, coerce[PNDArray](nd.pType).nDims, nd.pType.required)
      }
      case NDArrayMap2(l, r, lName, rName, body) => {
        // TODO: does body depend on env of l?
        // seems not based on Parser.scala line 669
        l.inferSetPType(env)
        body.inferSetPType(env)

        PNDArray(body.pType, coerce[PNDArray](l.pType).nDims, l.pType.required)
      }
      case NDArrayReindex(nd, indexExpr) => {
        nd.inferSetPType(env)

        PNDArray(coerce[PNDArray](nd.pType).elementType, indexExpr.length, nd.pType.required)
      }
      case NDArrayAgg(nd, axes) => {
        nd.inferSetPType(env)

        val childType = coerce[PNDArray](nd.pType)
        PNDArray(childType.elementType, childType.nDims - axes.length, childType.required)
      }
      case NDArrayRef(nd, idxs) => {
        nd.inferSetPType(env)

        // TODO: Need to bind? How will this be used concretely?
        val boundEnv = env.bindEval("nd", nd.pType)

        // TODO: PInt64 seems unnecessary for nearly all arrays in variant space, stop asserting
        var allRequired = true
        idxs.foreach( idx => {
          idx.inferSetPType(boundEnv)

          if(allRequired && !idx.pType.required) {
            allRequired = false
          }

          assert(idx.pType.isOfType(PInt64()) || idx.pType.isOfType(PInt32()))
        })

        // TODO: In standing with other similar cases, should we be mutating elementType, or copyStreamable?
        coerce[PNDArray](nd.pType).elementType.setRequired(nd.pType.required && allRequired)
      }
      case NDArraySlice(nd, slices) => {
        nd.inferSetPType(env)
        val childTyp = coerce[PNDArray](nd.pType)

        // TODO: do slices need nd env?
        slices.inferSetPType(env)
        val remainingDims = coerce[PTuple](slices.pType).types.filter(_.isInstanceOf[PTuple])
        PNDArray(childTyp.elementType, remainingDims.length)
      }
      case NDArrayMatMul(l, r) => {
        l.inferSetPType(env)
        r.inferSetPType(env)
        val lTyp = coerce[PNDArray](l.pType)
        val rTyp = coerce[PNDArray](r.pType)
        PNDArray(lTyp.elementType, PNDArray.matMulNDims(lTyp.nDims, rTyp.nDims), lTyp.required && rTyp.required)
      }
      case NDArrayWrite(_, _) => PVoid
      case AggFilter(cond, aggIR, isScan) => {
        cond.inferSetPType(env)
        // TODO: Does AggFilter need cond env? If so under what name do we bind it
        // I can imagine a case where the operation taken matters quite a lot
        // example: we are subtracting a series of numbers, and know our largest number is < 2^31
        aggIR.inferSetPType(env)
        aggIR.pType
      }
      case AggExplode(array, name, aggBody, isScan) => {
        array.inferSetPType(env)

        // TODO: how is this concretely different from bindEval?
        // TODO: Needed? We currently do a much simpler type check in TypeCheck, no env
        // also, what's the benefit of bindScan/agg over bindEval for type inference?
        // or should this be a BindingEnv?
        val boundEnv =
          if (isScan)
            env.bindScan(name, array.pType)
          else
            env.bindAgg(name, array.pType)

        aggBody.inferSetPType(boundEnv)

        aggBody.pType
      }
      case AggGroupBy(key, aggIR, isScan) => {
        // TODO: Check that aggIR doesn't need key pType; don't think so, Parser doesn't bind key env to aggIR
        key.inferSetPType(env)
        aggIR.inferSetPType(env)

        PDict(key.pType, aggIR.pType)
      }
      case AggArrayPerElement(a, elementName, indexName, aggBody, knownLength, isScan) => {
        a.inferSetPType(env)

        knownLength match {
          case Some(lengthIR) => lengthIR.inferSetPType(env)
        }
        // TODO: If we statically know the length, I would like to use that during inference
        val boundEnv =
          if (isScan)
            env.bindScan(elementName -> a.pType)
          else
            env.bindScan(elementName -> a.pType)
        aggBody.inferSetPType(boundEnv)
        PArray(aggBody.pType)
      }
      case ApplyAggOp(constructorArgs, initOpArgs, seqOpArgs, aggSig) => {
        constructorArgs.foreach(_.inferSetPType(env))

        initOpArgs match {
          case Some(opArgs) => opArgs.foreach(_.inferSetPType(env))
        }

        seqOpArgs.foreach(_.inferSetPType(env))

        // TODO: Something else?
        PType.canonical(AggOp.getType(aggSig))
      }
      case ApplyScanOp(constructorArgs, initOpArgs, seqOpArgs, aggSig) => {
        constructorArgs.foreach(_.inferSetPType(env))

        initOpArgs match {
          case Some(opArgs) => opArgs.foreach(_.inferSetPType(env))
        }

        seqOpArgs.foreach(_.inferSetPType(env))

        // TODO: Something else?
        PType.canonical(AggOp.getType(aggSig))
      }
      case MakeStruct(fields) => {
        PStruct(fields.map {
          case (name, a) => {
            a.inferSetPType(env)
            (name, a.pType)
          }
        }: _*)
      }
      case SelectFields(old, fields) => {
        old.inferSetPType(env)
        val tbs = coerce[PStruct](old.pType)
        tbs.select(fields.toFastIndexedSeq)._1
      }
      case InsertFields(old, fields, fieldOrder) => {
        old.inferSetPType(env)
        val tbs = coerce[PStruct](old.pType)

        // TODO: does this need old environment? No name
        val s = tbs.insertFields(fields.map(f =>  {
          f._2.inferSetPType(env)
          (f._1, f._2.pType)
        }))
        fieldOrder.map { fds =>
          assert(fds.length == s.size)
          PStruct(fds.map(f => f -> s.fieldType(f)): _*)
        }.getOrElse(s)
      }
      case GetField(o, name) => {
        o.inferSetPType(env)
        val t = coerce[PStruct](o.pType)
        if (t.index(name).isEmpty)
          throw new RuntimeException(s"$name not in $t")
        // TODO: Why is this not a copy?
        val fd = t.field(name).typ
        fd.setRequired(t.required && fd.required)
      }
      case MakeTuple(values) => {
        PTuple(values.map(v => {
          v.inferSetPType(env)
          v.pType
        }).toFastIndexedSeq)
      }
      case GetTupleElement(o, idx) => {
        o.inferSetPType(env)
        val t = coerce[PTuple](o.pType)
        assert(idx >= 0 && idx < t.size)
        val fd = t.types(idx)
        fd.setRequired(t.required && fd.required)
      }
      case CollectDistributedArray(contexts, globals, contextsName, globalsName, body) => {
        contexts.inferSetPType(env)
        globals.inferSetPType(env)

        body.inferSetPType(env.bindEval(contextsName -> contexts.pType, globalsName -> globals.pType))
        PArray(body.pType)
      }
      case ReadPartition(_, _, _, rowType) => PStream(PType.canonical(rowType))
    }
  }
}
