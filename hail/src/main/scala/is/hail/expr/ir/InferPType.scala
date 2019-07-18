package is.hail.expr.ir

import is.hail.expr.types.physical._
import is.hail.expr.types.virtual.{TArray, TNDArray, TTuple, TVoid}
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
  def apply(ir: IR, env: Env[PType]): PType = {
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
        val head = vit.next()
        head.inferSetPType(env)

        while(vit.hasNext) {
          val value = vit.next()
          value.inferSetPType(env)
          assert(head.pType2 == value.pType2, "Values in Coalesce must all be of the same type")
        }

        head.pType2
      }
      // TODO: Why is there a type (var _type) on Ref? Cache to avoid lookup? if so seems area for bugs
      case Ref(name, _) => {
        env.lookup(name)
      }
      case In(_, t) => PType.canonical(t)
      // TODO: need to descend into args?
      case MakeArray(_, t) => PType.canonical(t)
      case MakeStream(_, t) => PType.canonical(t)
      case MakeNDArray(data, shape, _) => {
        data.inferSetPType(env)
        shape.inferSetPType(env)

        // TODO: Is this wrong?
        val nElem = shape.pType2.asInstanceOf[PTuple].size

        // TODO: requiredeness?
        PNDArray(coerce[PArray](data.pType2).elementType, nElem)
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

        assert(cond.pType2 == PBoolean())

        if (cnsq.pType2 != altr.pType2)
          cnsq.pType2.deepOptional()
        else {
          assert(cnsq.pType2 isOfType altr.pType2)
          cnsq.pType2
        }
      }
      case Let(name, value, body) => {
        value.inferSetPType(env)
        body.inferSetPType(env.bind(name, value.pType2))
        body.pType2
      }
      // TODO: It feels like this PType could benefit from inspecting the values it applies the op to
      // whenever the values are known at time of compilation (current only time of inference)
      case ApplyBinaryPrimOp(op, l, r) => {
          l.inferSetPType(env)
          r.inferSetPType(env)

          val required = l.pType2.required && r.pType2.required
          val vType = BinaryOp.getReturnType(op, l.pType2.virtualType, r.pType2.virtualType).setRequired(required)

          PType.canonical(vType)
      }
      case ApplyUnaryPrimOp(op, v) => {
        v.inferSetPType(env)
        println("IN")
        println(v.pType2)
        println(v.pType2.required)
        PType.canonical(UnaryOp.getReturnType(op, v.pType2.virtualType).setRequired(v.pType2.required))
      }
      case ApplyComparisonOp(op, l, r) => {
        l.inferSetPType(env)
        r.inferSetPType(env)
        // TODO: should this be l.pType2 == r.pType2 or isOfType
        assert(l.pType2 isOfType r.pType2)
        op match {
          case _: Compare => PInt32(l.pType2.required && r.pType2.required)
          case _ => PBoolean(l.pType2.required && r.pType2.required)
        }
      }
      case a: ApplyIR => {
        a.explicitNode.inferSetPType(env)
        a.explicitNode.pType2
      }
      case a: AbstractApplyNode[_] => {
        val args = a.args.map( ir => {
          ir.inferSetPType(env)
          ir
        })
        a.implementation.returnPType(args.map(_.pType2))
      }
      case _: Uniroot => PFloat64()
      // TODO: check that i.pType2 isOfType is correct (for ArrayRef, ArraySort)
      case ArrayRef(a, i) => {
        a.inferSetPType(env)
        i.inferSetPType(env)
        println("GOT AN I")
        println(i)
        println(i.pType2)
        assert(i.pType2 isOfType PInt32() )
        coerce[PStreamable](a.pType2).elementType.setRequired(a.pType2.required && i.pType2.required)
      }
      case ArraySort(a, leftName, rightName, compare) => {
        a.inferSetPType(env)
        val et = coerce[PStreamable](a.pType2).elementType

        compare.inferSetPType(env.bind(leftName -> et, rightName -> et))
        assert(compare.pType2.isOfType(PBoolean()))

        println("GOT A TYPE")
        println(compare.pType2)

        PArray(et, a.pType2.required)
      }
      case ToSet(a) => {
        a.inferSetPType(env)
        val et = coerce[PIterable](a.pType2).elementType
        PSet(et, a.pType2.required)
      }
      case ToDict(a) => {
        a.inferSetPType(env)
        val elt = coerce[PBaseStruct](coerce[PIterable](a.pType2).elementType)
        PDict(elt.types(0), elt.types(1), a.pType2.required)
      }
      case ToArray(a) => {
        a.inferSetPType(env)
        val elt = coerce[PIterable](a.pType2).elementType
        PArray(elt, a.pType2.required)
      }
      case ToStream(a) => {
        a.inferSetPType(env)
        val elt = coerce[PIterable](a.pType2).elementType
        PStream(elt, a.pType2.required)
      }
      case GroupByKey(collection) => {
        collection.inferSetPType(env)
        val elt = coerce[PBaseStruct](coerce[PStreamable](collection.pType2).elementType)
        PDict(elt.types(0), PArray(elt.types(1)), collection.pType2.required)
      }
      // TODO: Check the env.bindEval needs name, a
      case ArrayMap(a, name, body) => {
        println("IN ARRAY MAP")
        println(a)
        println(name)
        println(body)
        println("DONE WITH BODY")
        // infer array side of tree fully
        a.inferSetPType(env)
        println("past infer a")
        println(a.pType2)
        println("that is a.pType2")
        // push do the element side, applies to each element
        body.inferSetPType(env.bind(name, a.pType2))
        // TODO: why setRequired false?
        coerce[PStreamable](a.pType2).copyStreamable(body.pType2.setRequired(false))
      }
      case ArrayFilter(a, name, cond) => {
        a.inferSetPType(env)
        a.pType2
      }
      // TODO: Check the env.bindEval needs name, a
      case ArrayFlatMap(a, name, body) => {
        a.inferSetPType(env)
        body.inferSetPType(env.bind(name, a.pType2))
        coerce[PStreamable](a.pType2).copyStreamable(coerce[PIterable](body.pType2).elementType)
      }
      case ArrayFold(a, zero, accumName, valueName, body) => {
        a.inferSetPType(env)
        zero.inferSetPType(env)

        body.inferSetPType(env.bind(accumName -> zero.pType2, valueName -> a.pType2))

        assert(body.pType2 == zero.pType2)
        zero.pType2
      }
      case ArrayScan(a, zero, accumName, valueName, body) => {
        a.inferSetPType(env)
        zero.inferSetPType(env)

        body.inferSetPType(env.bind(accumName -> zero.pType2, valueName -> a.pType2))

        assert(body.pType2 == zero.pType2)

        coerce[PStreamable](a.pType2).copyStreamable(zero.pType2)
      }
      case ArrayLeftJoinDistinct(lIR, rIR, lName, rName, compare, join) => {
        lIR.inferSetPType(env)
        rIR.inferSetPType(env)

        assert(lIR.pType2 == rIR.pType2)

        // TODO: Does left / right need same type? seems yes
        // TODO: if so, join can depend on only on left
        join.inferSetPType(env.bind(lName, lIR.pType2))


        // TODO: Not sure why InferType used copyStreamable here
        PArray(join.pType2)
      }
      case NDArrayShape(nd) => {
        nd.inferSetPType(env)
        PTuple(IndexedSeq.tabulate(nd.pType2.asInstanceOf[PNDArray].nDims)(_ => PInt64()), nd.pType2.required)
      }
      case NDArrayReshape(nd, shape) => {
        nd.inferSetPType(env)
        shape.inferSetPType(env)

        PNDArray(coerce[PNDArray](nd.pType2).elementType, shape.pType2.asInstanceOf[TTuple].size, nd.pType2.required)
      }
      case NDArrayMap(nd, name, body) => {
        nd.inferSetPType(env)
        body.inferSetPType(env.bind(name, nd.pType2))

        PNDArray(body.pType2, coerce[PNDArray](nd.pType2).nDims, nd.pType2.required)
      }
      case NDArrayMap2(l, r, lName, rName, body) => {
        // TODO: does body depend on env of l?
        // seems not based on Parser.scala line 669
        l.inferSetPType(env)
        body.inferSetPType(env)

        PNDArray(body.pType2, coerce[PNDArray](l.pType2).nDims, l.pType2.required)
      }
      case NDArrayReindex(nd, indexExpr) => {
        nd.inferSetPType(env)

        PNDArray(coerce[PNDArray](nd.pType2).elementType, indexExpr.length, nd.pType2.required)
      }
      case NDArrayRef(nd, idxs) => {
        nd.inferSetPType(env)

        // TODO: Need to bind? How will this be used concretely?
        val boundEnv = env.bind("nd", nd.pType2)

        // TODO: PInt64 seems unnecessary for nearly all arrays in variant space, stop asserting
        var allRequired = true
        idxs.foreach( idx => {
          idx.inferSetPType(boundEnv)

          if(allRequired && !idx.pType2.required) {
            allRequired = false
          }

          assert(idx.pType2.isOfType(PInt64()) || idx.pType2.isOfType(PInt32()))
        })

        // TODO: In standing with other similar cases, should we be mutating elementType, or copyStreamable?
        coerce[PNDArray](nd.pType2).elementType.setRequired(nd.pType2.required && allRequired)
      }
      case NDArraySlice(nd, slices) => {
        nd.inferSetPType(env)
        val childTyp = coerce[PNDArray](nd.pType2)

        // TODO: do slices need nd env?
        slices.inferSetPType(env)
        val remainingDims = coerce[PTuple](slices.pType2).types.filter(_.isInstanceOf[PTuple])
        PNDArray(childTyp.elementType, remainingDims.length)
      }
      case NDArrayMatMul(l, r) => {
        l.inferSetPType(env)
        r.inferSetPType(env)
        val lTyp = coerce[PNDArray](l.pType2)
        val rTyp = coerce[PNDArray](r.pType2)
        PNDArray(lTyp.elementType, TNDArray.matMulNDims(lTyp.nDims, rTyp.nDims), lTyp.required && rTyp.required)
      }
      case NDArrayWrite(_, _) => PVoid
      case MakeStruct(fields) => {
        PStruct(fields.map {
          case (name, a) => {
            a.inferSetPType(env)
            (name, a.pType2)
          }
        }: _*)
      }
      case SelectFields(old, fields) => {
        old.inferSetPType(env)
        val tbs = coerce[PStruct](old.pType2)
        tbs.select(fields.toFastIndexedSeq)._1
      }
      case InsertFields(old, fields, fieldOrder) => {
        old.inferSetPType(env)
        val tbs = coerce[PStruct](old.pType2)

        // TODO: does this need old environment? No name
        val s = tbs.insertFields(fields.map(f =>  {
          f._2.inferSetPType(env)
          (f._1, f._2.pType2)
        }))
        fieldOrder.map { fds =>
          assert(fds.length == s.size)
          PStruct(fds.map(f => f -> s.fieldType(f)): _*)
        }.getOrElse(s)
      }
      case GetField(o, name) => {
        o.inferSetPType(env)
        val t = coerce[PStruct](o.pType2)
        if (t.index(name).isEmpty)
          throw new RuntimeException(s"$name not in $t")
        // TODO: Why is this not a copy?
        val fd = t.field(name).typ
        fd.setRequired(t.required && fd.required)
      }
      case MakeTuple(values) => {
        PTuple(values.map(v => {
          v.inferSetPType(env)
          v.pType2
        }).toFastIndexedSeq)
      }
      case GetTupleElement(o, idx) => {
        o.inferSetPType(env)
        val t = coerce[PTuple](o.pType2)
        assert(idx >= 0 && idx < t.size)
        val fd = t.types(idx)
        fd.setRequired(t.required && fd.required)
      }
      case CollectDistributedArray(contexts, globals, contextsName, globalsName, body) => {
        contexts.inferSetPType(env)
        globals.inferSetPType(env)

        body.inferSetPType(env.bind(contextsName -> contexts.pType2, globalsName -> globals.pType2))
        PArray(body.pType2)
      }
      case ReadPartition(_, _, _, rowType) => PStream(PType.canonical(rowType))
      case _ => PVoid
//      case ApplySpecial(name, irs) => {
//        println("Name of applyspecial")
//        println(name)
//        val it = irs.iterator
//        val head = it.next()
//        head.inferSetPType(env)
//
//        while(it.hasNext) {
//          val value = it.next()
//
//          value.inferSetPType(env)
//          println("value.pType2")
//          println(value.pType2)
//          println("head.pType2")
//          println(head.pType2)
//          assert(value.pType2 == head.pType2)
//        }
//
//        head.pType2
//      }
//      case _: ArrayAggScan => PVoid
//      case _: ArrayAgg => PVoid
    }
  }
}
