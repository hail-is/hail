package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations.BroadcastRow
import is.hail.expr._
import is.hail.expr.types._
import is.hail.utils._
import org.apache.spark.sql.Row

object PruneDeadFields {
  def isSupertype(superType: BaseType, subType: BaseType): Boolean = {
    try {
      (superType, subType) match {
        case (tt1: TableType, tt2: TableType) =>
          isSupertype(tt1.globalType, tt2.globalType) &&
            isSupertype(tt1.rowType, tt2.rowType)
        case (mt1: MatrixType, mt2: MatrixType) =>
          isSupertype(mt1.globalType, mt2.globalType) &&
            isSupertype(mt1.rowType, mt2.rowType) &&
            isSupertype(mt1.colType, mt2.colType) &&
            isSupertype(mt1.entryType, mt2.entryType)
        case (TArray(et1, r1), TArray(et2, r2)) => r1 == r2 && isSupertype(et1, et2)
        case (TSet(et1, r1), TSet(et2, r2)) => r1 == r2 && isSupertype(et1, et2)
        case (TDict(kt1, vt1, r1), TDict(kt2, vt2, r2)) => r1 == r2 && isSupertype(kt1, kt2) && isSupertype(vt1, vt2)
        case (s1: TStruct, s2: TStruct) =>
          var idx = -1
          s1.required == s2.required && s1.fields.forall { f =>
            val s2field = s2.field(f.name)
            if (s2field.index > idx) {
              idx = s2field.index
              isSupertype(f.typ, s2field.typ)
            } else
              false
          }
        case (agg1: TAggregable, agg2: TAggregable) => isSupertype(agg1.elementType, agg2.elementType)
        case (t1: Type, t2: Type) => t1 == t2
        case _ => fatal(s"invalid comparison: $superType / $subType")
      }
    } catch {
      case e: Throwable =>
        fatal(s"error while checking subtype:\n  super: $superType\n  sub:   $subType", e)
    }
  }

  def apply(ir: BaseIR): BaseIR = {

    try {
      ir.deepCopy() match {
        case x: MatrixIR => rebuild(x, x.typ)
        case x: TableIR => rebuild(x, x.typ)
        case x: IR => rebuild(x)
      }
    } catch {
      case e: Throwable => fatal(s"error trying to rebuild IR:\n${ Pretty(ir) }", e)
    }
  }


  def minimal(tt: TableType): TableType = {
    val keySet = tt.key.iterator.flatten.toSet
    tt.copy(
      rowType = tt.rowType.filter(keySet)._1,
      globalType = TStruct(tt.globalType.required)
    )
  }

  def minimal(mt: MatrixType): MatrixType = {
    val rowKeySet = mt.rowKey.toSet
    val colKeySet = mt.colKey.toSet
    mt.copy(
      rvRowType = mt.rvRowType.filter(rowKeySet ++ Array(MatrixType.entriesIdentifier))._1,
      colType = mt.colType.filter(colKeySet)._1,
      globalType = TStruct(mt.globalType.required)
    )
  }

  def minimal[T <: Type](base: T): T = {
    val result = base match {
      case ts: TStruct => TStruct(ts.required)
      case ta: TArray => TArray(minimal(ta.elementType), ta.required)
      case tagg: TAggregable => tagg.copy(elementType = minimal(tagg.elementType))
      case t => t
    }
    result.asInstanceOf[T]
  }

  def unionBaseType[T <: BaseType](base: T, children: T*): T = {
    val r = base match {
      case tt: TableType =>
        val ttChildren = children.map(_.asInstanceOf[TableType])
        tt.copy(
          rowType = unionType(tt.rowType, ttChildren.map(_.rowType): _*),
          globalType = unionType(tt.globalType, ttChildren.map(_.globalType): _*)
        )
      case mt: MatrixType =>
        val mtChildren = children.map(_.asInstanceOf[MatrixType])
        mt.copy(
          globalType = unionType(mt.globalType, mtChildren.map(_.globalType): _*),
          rvRowType = unionType(mt.rvRowType, mtChildren.map(_.rvRowType): _*),
          colType = unionType(mt.colType, mtChildren.map(_.colType): _*)
        )
    }
    r.asInstanceOf[T]
  }

  def unionType[T <: Type](base: T, children: T*): T = {
    children.foreach(t => assert(isSupertype(t, base), s"incompatibility:\n  ${ t.parsableString() }\n  ${ base.parsableString }"))

    if (children.isEmpty)
      return minimal(base)

    val r = base match {
      case ts: TStruct =>
        val subStructs = children.map(_.asInstanceOf[TStruct])
        val subFields = ts.fields.map { f =>
          f -> subStructs.flatMap(s => s.fieldOption(f.name))
        }
          .filter(_._2.nonEmpty)
          .map { case (f, ss) => f.name -> unionType(f.typ, ss.map(_.typ): _*) }
        TStruct(ts.required, subFields: _*)
      case ta: TArray =>
        TArray(unionType(ta.elementType, children.map(_.asInstanceOf[TArray].elementType): _*), ta.required)
      case tagg: TAggregable => tagg.copy(elementType =
        unionType(tagg.elementType, children.map(_.asInstanceOf[TAggregable].elementType): _*))
      case _ =>
        assert(children.forall(_ == base))
        base
    }
    r.asInstanceOf[T]
  }

  def unify(base: Env[Type], envs: Env[Type]*): Env[Type] = {
    val bindings = base.m.toArray.map { case (name, baseType) =>
      name -> unionType[Type](baseType, envs.flatMap(_.lookupOption(name)): _*)
    }
    new Env[Type].bind(bindings: _*)
  }

  def baseTypeToEnv(bt: BaseType): Env[Type] = {
    bt match {
      case tt: TableType =>
        Env.empty[Type]
          .bind("row", tt.rowType)
          .bind("global", tt.globalType)
          .bind("AGG", tt.tAgg)
      case mt: MatrixType =>
        Env.empty[Type]
          .bind("global", mt.globalType)
          .bind("sa", mt.colType)
          .bind("va", mt.rvRowType)
          .bind("g", mt.entryType)
          .bind("AGG", mt.rowEC.st("AGG")._2)
    }
  }

  def getDeps(ir: IR, rType: Type, base: TableType): (TableType, Memo[Type]) = {
    val env = baseTypeToEnv(base)
    val memo = Memo.empty[Type]
    val depEnv = getDeps(ir, rType, env, memo)
    val min = minimal(base)
    val rowArgs = (Iterator.single(min.rowType) ++
      depEnv.lookupOption("row").map(_.asInstanceOf[TStruct]).iterator ++
      depEnv.lookupOption("AGG").map(_.asInstanceOf[TAggregable].elementType.asInstanceOf[TStruct]).iterator).toArray
    val globalArgs = (Iterator.single(min.globalType) ++ depEnv.lookupOption("global").map(_.asInstanceOf[TStruct]).iterator).toArray
    base.copy(rowType = unionType(base.rowType, rowArgs: _*),
      globalType = unionType(base.globalType, globalArgs: _*)) -> memo
  }

  def getDeps(ir: IR, rType: Type, base: MatrixType): (MatrixType, Memo[Type]) = {
    val env = baseTypeToEnv(base)
    val memo = Memo.empty[Type]
    val depEnv = getDeps(ir, rType, env, memo)
    val min = minimal(base)
    val eField = base.rvRowType.field(MatrixType.entriesIdentifier)
    val rowArgs = (Iterator.single(min.rvRowType) ++ depEnv.lookupOption("va").iterator ++
      Iterator.single(TStruct(eField.name -> TArray(
        unionType(eField.typ.asInstanceOf[TArray].elementType,
          (depEnv.lookupOption("g").iterator ++ depEnv.lookupOption("AGG").map(_.asInstanceOf[TAggregable].elementType).iterator).toArray: _*),
        eField.typ.required)))).toArray
    val colArgs = (Iterator.single(min.colType) ++ depEnv.lookupOption("sa").iterator).toArray
    val globalArgs = (Iterator.single(min.globalType) ++ depEnv.lookupOption("global").iterator).toArray
    base.copy(rvRowType = unionType(base.rvRowType, rowArgs: _*).asInstanceOf[TStruct],
      globalType = unionType(base.globalType, globalArgs: _*).asInstanceOf[TStruct],
      colType = unionType(base.colType, colArgs: _*).asInstanceOf[TStruct]) -> memo
  }

  def getDeps(ir: IR, rType: Type, base: Env[Type], memo: Memo[Type]): Env[Type] = {
    memo.bind(ir, rType)
    ir match {
      case I32(_) => Env.empty[Type]
      case I64(_) => Env.empty[Type]
      case F32(_) => Env.empty[Type]
      case F64(_) => Env.empty[Type]
      case Str(_) => Env.empty[Type]
      case True() => Env.empty[Type]
      case False() => Env.empty[Type]
      case Cast(v, t) => getDeps(v, v.typ, base, memo)
      case NA(typ) => Env.empty[Type]
      case IsNA(value) => getDeps(value, minimal(value.typ), base, memo)
      case If(cond, cnsq, alt) =>
        unify(base,
          getDeps(cond, cond.typ, base, memo),
          getDeps(cnsq, rType, base, memo),
          getDeps(alt, rType, base, memo)
        )
      case Let(name, value, body) =>
        val min = minimal(value.typ)
        val bodyEnv = getDeps(body, rType, base.bind(name, min), memo)
        val valueType = bodyEnv.lookupOption(name).getOrElse(min)
        unify(base,
          bodyEnv.delete(name),
          getDeps(value, valueType, base, memo)
        )
      case Ref(name, t) =>
        assert(isSupertype(rType, t), s"not subtype:\n  ret:  ${ rType.parsableString() }\n  base: ${ t.parsableString() }")
        Env.empty[Type].bind(name, rType)
      case ApplyBinaryPrimOp(_, l, r) =>
        unify(base,
          getDeps(l, l.typ, base, memo),
          getDeps(r, r.typ, base, memo))
      case ApplyUnaryPrimOp(_, x) =>
        getDeps(x, x.typ, base, memo)
      case MakeArray(args, _) =>
        val eltType = rType.asInstanceOf[TArray].elementType
        unify(base, args.map(a => getDeps(a, eltType, base, memo)): _*)
      case ArrayRef(a, i) =>
        unify(base, getDeps(a, a.typ.asInstanceOf[TArray].copy(elementType = rType), base, memo), getDeps(i, i.typ, base, memo))
      case ArrayLen(a) =>
        getDeps(a, minimal(a.typ), base, memo)
      case ArrayRange(start, stop, step) =>
        unify(base,
          getDeps(start, start.typ, base, memo),
          getDeps(stop, start.typ, base, memo),
          getDeps(step, step.typ, base, memo))
      case ArrayMap(a, name, body) =>
        val aType = a.typ.asInstanceOf[TArray]
        val bodyEnv = getDeps(body,
          rType.asInstanceOf[TArray].elementType,
          base.bind(name, -aType.elementType),
          memo)
        val valueType = bodyEnv.lookupOption(name).getOrElse(minimal(-aType.elementType))
        unify(base,
          bodyEnv.delete(name),
          getDeps(a, aType.copy(elementType = valueType), base, memo)
        )
      case ArrayFilter(a, name, cond) =>
        val aType = a.typ.asInstanceOf[TArray]
        val bodyEnv = getDeps(cond, cond.typ, base.bind(name, -aType.elementType), memo)
        val valueType = bodyEnv.lookupOption(name).getOrElse(minimal(-aType.elementType))
        unify(base,
          bodyEnv.delete(name),
          getDeps(a, aType.copy(elementType = valueType), base, memo)
        )
      case ArrayFlatMap(a, name, body) =>
        val aType = a.typ.asInstanceOf[TArray]
        val bodyEnv = getDeps(body, rType, base.bind(name, -aType.elementType), memo)
        val valueType = bodyEnv.lookupOption(name).getOrElse(minimal(-aType.elementType))
        unify(base,
          bodyEnv.delete(name),
          getDeps(a, aType.copy(elementType = valueType), base, memo)
        )
      case ArrayFold(a, zero, accumName, valueName, body) =>
        assert(rType == zero.typ)
        val aType = a.typ.asInstanceOf[TArray]
        val zeroEnv = getDeps(zero, zero.typ, base, memo)
        val bodyEnv = getDeps(body,
          zero.typ,
          base.bind(accumName -> zero.typ, valueName -> -aType.elementType),
          memo)
        val valueType = bodyEnv.lookupOption(valueName).getOrElse(minimal(-aType.elementType))
        unify(base,
          zeroEnv,
          bodyEnv.delete(accumName).delete(valueName),
          getDeps(a, aType.copy(elementType = valueType), base, memo)
        )
      case ArrayFor(a, valueName, body) =>
        val aType = a.typ.asInstanceOf[TArray]
        val bodyEnv = getDeps(body,
          body.typ,
          base.bind(valueName -> -aType.elementType),
          memo)
        val valueType = bodyEnv.lookupOption(valueName).getOrElse(minimal(-aType.elementType))
        unify(base,
          bodyEnv.delete(valueName),
          getDeps(a, aType.copy(elementType = valueType), base, memo)
        )
      case ApplyAggOp(a, constructorArgs, initOpArgs, _) =>
        unify(base,
          Seq(getDeps(a, a.typ, base, memo)) ++
            (constructorArgs ++ initOpArgs.toSeq.flatten).map(arg => getDeps(arg, arg.typ, base, memo)): _*
        )
      case SeqOp(a, i, _) => unify(base, getDeps(a, a.typ, base, memo), getDeps(i, i.typ, base, memo))
      case Begin(xs) => unify(base, xs.map(x => getDeps(x, x.typ, base, memo)): _*)
      case MakeStruct(fields) =>
        val sType = rType.asInstanceOf[TStruct]
        unify(base, fields.flatMap { case (fname, fir) =>
          // ignore unreachable fields, these are eliminated on the upwards pass
          sType.fieldOption(fname).map(f => getDeps(fir, f.typ, base, memo))
        }: _*)
      case InsertFields(old, fields) =>
        val sType = rType.asInstanceOf[TStruct]
        val insFieldNames = fields.map(_._1).toSet
        val rightDep = sType.filter(f => insFieldNames.contains(f.name))._1
        val rightDepFields = rightDep.fieldNames.toSet
        val leftDep = TStruct(old.typ.required,
          old.typ.asInstanceOf[TStruct]
            .fields
            .flatMap { f =>
              if (rightDep.hasField(f.name))
                Some(f.name -> minimal(f.typ))
              else
                sType.fieldOption(f.name).map(f.name -> _.typ)
            }: _*)
        unify(base,
          Seq(getDeps(old, leftDep, base, memo)) ++
            // ignore unreachable fields, these are eliminated on the upwards pass
            fields.flatMap { case (fname, fir) =>
              rightDep.fieldOption(fname).map(f => getDeps(fir, f.typ, base, memo))
            }: _*
        )
      case x@GetField(o, name) =>
        getDeps(o, o.typ.asInstanceOf[TStruct].filter(_.name == name)._1, base, memo)
      case x@MakeTuple(types) =>
        val tType = rType.asInstanceOf[TTuple]
        assert(types.length == tType.size)
        unify(
          base,
          types.zip(tType.types).map { case (tir, t) => getDeps(tir, t, base, memo) }: _*
        )
      case GetTupleElement(o, idx) =>
        // FIXME handle tuples better
        val childTupleType = o.typ.asInstanceOf[TTuple]
        val tupleDep = TTuple(childTupleType.required,
          childTupleType.types
            .zipWithIndex
            .map { case (t, i) => if (i == idx) rType else minimal(t) }: _*)
        getDeps(o, tupleDep, base, memo)
      case In(i, _) => Env.empty[Type]
      case Die(_) => Env.empty[Type]
      case Apply(function, args) =>
        // assume everything is required
        unify(
          base,
          args.map(a => getDeps(a, a.typ, base, memo)): _*
        )
      case ApplySpecial(function, args) =>
        // assume everything is required
        unify(
          base,
          args.map(a => getDeps(a, a.typ, base, memo)): _*
        )
    }
  }

  def rebuild(tir: TableIR, dep: TableType): TableIR = {
    assert(isSupertype(dep, tir.typ), s"not subtype: \n  $tir\n  ${ tir.typ }\n  $dep")
    tir match {
      case x@TableRead(_, _, _) => upcast(x, dep)
      case x@TableLiteral(_) => x
      case x@TableParallelize(_, _, _) => x
      case x@TableImport(paths, typ, readerOpts) =>
        val fieldsToRead = readerOpts.originalType.fields.flatMap(f => dep.rowType.fieldOption(f.name).map(_ => f.index)).toArray
        val newTyp = typ.copy(rowType = TStruct(typ.rowType.required,
          fieldsToRead.map(i => readerOpts.originalType.fieldNames(i) -> readerOpts.originalType.types(i)): _*))
        TableImport(paths, newTyp, readerOpts.copy(useColIndices = fieldsToRead))
      case x@TableRange(_, _) => x
      case x@TableJoin(left, right, joinType) =>
        val depFields = dep.rowType.fieldNames.toSet
        val leftDep = left.typ.copy(
          rowType = left.typ.rowType.filter(f => depFields.contains(f.name))._1,
          globalType = dep.globalType)
        val rightDep = right.typ.copy(
          rowType = right.typ.rowType.filter(f => depFields.contains(x.rightFieldMapping(f.name)))._1,
          globalType = minimal(dep.globalType))
        TableJoin(rebuild(left, leftDep), rebuild(right, rightDep), joinType)
      case x@TableExplode(child, field) =>
        val minChild = minimal(child.typ)
        val dep2 = unionBaseType(child.typ, dep.copy(rowType = dep.rowType.filter(_.name != field)._1),
          minChild.copy(rowType = unionType(
            child.typ.rowType, minChild.rowType, child.typ.rowType.filter(_.name == field)._1)))
        TableExplode(rebuild(child, dep2), field)
      case x@TableFilter(child, pred) =>
        val (irDep, memo) = getDeps(pred, pred.typ, child.typ)
        val child2 = rebuild(child, unionBaseType(child.typ, dep, irDep))
        val pred2 = rebuildIR(pred, child2.typ, memo)
        TableFilter(child2, pred2)
      case x@TableKeyBy(child, keys, nPartitionKeys, sort) =>
        TableKeyBy(rebuild(child, dep.copy(key = child.typ.key)), keys, nPartitionKeys, sort)
      case x@TableMapRows(child, newRow, newKey) =>
        val (rowDep, memo) = getDeps(newRow, dep.rowType, child.typ)
        val child2 = rebuild(child, unionBaseType(child.typ, dep.copy(rowType = rowDep.rowType, key = child.typ.key), rowDep))
        TableMapRows(child2, rebuildIR(newRow, child2.typ, memo), newKey)
      case x@TableMapGlobals(child, newRow, value) =>
        val (globalDep, memo) = getDeps(newRow, dep.globalType, child.typ)
        // fixme push down into value
        val child2 = rebuild(child, unionBaseType(child.typ, dep.copy(globalType = globalDep.globalType), globalDep))
        TableMapGlobals(child2, rebuildIR(newRow, child2.typ, memo, "value" -> value.t), value)
      case x@MatrixColsTable(child) =>
        val minChild = minimal(child.typ)
        val mtDep = minChild.copy(
          globalType = dep.globalType,
          colType = dep.rowType)
        MatrixColsTable(rebuild(child, mtDep))
      case x@MatrixRowsTable(child) =>
        val minChild = minimal(child.typ)
        val mtDep = minChild.copy(
          globalType = dep.globalType,
          rvRowType = unionType(child.typ.rvRowType, minChild.rvRowType, dep.rowType))
        MatrixRowsTable(rebuild(child, mtDep))
      case x@MatrixEntriesTable(child) =>
        val mtDep = child.typ.copy(
          globalType = dep.globalType,
          colType = TStruct(child.typ.colType.required,
            child.typ.colType.fields.flatMap(f => dep.rowType.fieldOption(f.name).map(f2 => f.name -> f2.typ)): _*),
          rvRowType = TStruct(child.typ.rvRowType.required, child.typ.rvRowType.fields.flatMap { f =>
            if (f.name == MatrixType.entriesIdentifier) {
              Some(f.name -> TArray(TStruct(child.typ.entryType.required, child.typ.entryType.fields.flatMap { entryField =>
                dep.rowType.fieldOption(entryField.name).map(f2 => f2.name -> f2.typ)
              }: _*), f.typ.required))
            } else {
              dep.rowType.fieldOption(f.name).map(f2 => f.name -> f2.typ)
            }
          }: _*
          ))
        val child2 = rebuild(child, mtDep)
        MatrixEntriesTable(child2)
      case x@TableUnion(children) =>
        TableUnion(children.map(rebuild(_, dep)))
    }
  }

  def rebuild(mir: MatrixIR, dep: MatrixType): MatrixIR = {
    assert(isSupertype(dep, mir.typ), s"not subtype: \n  $mir\n  ${ mir.typ }\n  $dep")
    mir match {
      case x@FilterColsIR(child, pred) =>
        val (irDep, memo) = getDeps(pred, pred.typ, child.typ)
        val child2 = rebuild(child, unionBaseType(child.typ, dep, irDep))
        FilterColsIR(child2, rebuildIR(pred, child2.typ, memo))
      case x@MatrixFilterRowsIR(child, pred) =>
        val (irDep, memo) = getDeps(pred, pred.typ, child.typ)
        val child2 = rebuild(child, unionBaseType(child.typ, dep, irDep))
        MatrixFilterRowsIR(child2, rebuildIR(pred, child2.typ, memo))
      case x@MatrixFilterEntries(child, pred) =>
        val (irDep, memo) = getDeps(pred, pred.typ, child.typ)
        val child2 = rebuild(child, unionBaseType(child.typ, dep, irDep))
        MatrixFilterEntries(child2, rebuildIR(pred, child2.typ, memo))
      case x@MatrixMapEntries(child, newEntries) =>
        val (irDep, memo) = getDeps(newEntries, dep.entryType, child.typ)
        val depMod = dep.copy(rvRowType = TStruct(dep.rvRowType.required, dep.rvRowType.fields.map { f =>
          if (f.name == MatrixType.entriesIdentifier)
            f.name -> f.typ.asInstanceOf[TArray].copy(elementType = irDep.entryType)
          else
            f.name -> f.typ
        }: _*))
        val child2 = rebuild(child, unionBaseType(child.typ, depMod, irDep))
        MatrixMapEntries(child2, rebuildIR(newEntries, child2.typ, memo))
      case x@MatrixMapRows(child, newRow, newKey) =>
        val (irDep, memo) = getDeps(newRow, dep.rowType, child.typ)
        val depMod = dep.copy(rvRowType = TStruct(irDep.rvRowType.fields.map { f =>
          if (f.name == MatrixType.entriesIdentifier)
            f.name -> unionType(child.typ.rvRowType.field(MatrixType.entriesIdentifier).typ,
              f.typ,
              dep.rvRowType.field(MatrixType.entriesIdentifier).typ)
          else
            f.name -> f.typ
        }: _*), rowKey = child.typ.rowKey)
        val child2 = rebuild(child, unionBaseType(child.typ, depMod, irDep))
        MatrixMapRows(child2, rebuildIR(newRow, child2.typ, memo), newKey)
      case x@MatrixMapCols(child, newCol, newKey) =>
        val (irDep, memo) = getDeps(newCol, dep.colType, child.typ)
        val child2 = rebuild(child, unionBaseType(child.typ, dep.copy(
          colType = irDep.colType,
          colKey = child.typ.colKey
        ), irDep))
        MatrixMapCols(child2, rebuildIR(newCol, child2.typ, memo), newKey)
      case x@MatrixMapGlobals(child, newRow, value) =>
        val (irDep, memo) = getDeps(newRow, dep.globalType, child.typ)
        // fixme push down into value
        val child2 = rebuild(child, unionBaseType(child.typ, dep.copy(globalType = irDep.globalType), irDep))
        MatrixMapGlobals(child2, rebuildIR(newRow, child2.typ, memo, "value" -> value.t), value)
      case x@MatrixRead(_, _, _, _, _) => upcast(x, dep)
      case x@MatrixLiteral(typ, value) => x
      case x@FilterCols(child, cond) =>
        FilterCols(rebuild(child, child.typ), cond)
      case x@MatrixFilterRowsAST(child, cond) =>
        MatrixFilterRowsAST(rebuild(child, child.typ), cond)
      case x@ChooseCols(child, oldIndices) =>
        ChooseCols(rebuild(child, dep), oldIndices)
      case x@CollectColsByKey(child) =>
        val colKeySet = dep.colKey.toSet
        val explodedDep = dep.copy(
          colType = TStruct(dep.colType.required, dep.colType.fields.map { f =>
            if (colKeySet.contains(f.name))
              f.name -> f.typ
            else {
              f.name -> f.typ.asInstanceOf[TArray].elementType
            }
          }: _*),
          rvRowType = dep.rvRowType.copy(fields = dep.rvRowType.fields.map { f =>
            if (f.name == MatrixType.entriesIdentifier)
              f.copy(typ = TArray(
                TStruct(dep.entryType.required, dep.entryType.fields.map(ef =>
                  ef.name -> ef.typ.asInstanceOf[TArray].elementType): _*), f.typ.required))
            else
              f
          })
        )
        CollectColsByKey(rebuild(child, explodedDep))
    }
  }

  def rebuild(ir: IR): IR = {
    ir match {
      case TableCount(child) => TableCount(rebuild(child, minimal(child.typ)))
      case TableWrite(child, path, overwrite, spec) => TableWrite(rebuild(child, child.typ), path, overwrite, spec)
      case TableExport(child, path, typesFile, header, exportType) =>
        TableExport(rebuild(child, child.typ), path, typesFile, header, exportType)
      case TableAggregate(child, query) =>
        val (queryDep, memo) = getDeps(query, query.typ, child.typ)
        val child2 = rebuild(child, queryDep)
        val query2 = rebuildIR(query, child2.typ, memo)
        TableAggregate(child2, query2)
      case MatrixWrite(child, f) =>
        MatrixWrite(rebuild(child, child.typ), f)
      case x => Copy(x, x.children.map(c => rebuild(c.asInstanceOf[IR]))).asInstanceOf[IR]
    }
  }

  def rebuildIR(ir: IR, in: BaseType, memo: Memo[Type], bindings: (String, Type)*): IR = {
    rewriteIR(ir, baseTypeToEnv(in).bind(bindings: _*), memo)
  }

  def rewriteIR(ir: IR, in: Env[Type], memo: Memo[Type]): IR = {
    val dep = memo.lookup(ir)
    ir match {
      case I32(_) => ir
      case I64(_) => ir
      case F32(_) => ir
      case F64(_) => ir
      case Str(_) => ir
      case True() => ir
      case False() => ir
      case Cast(v, t) =>
        assert(dep == t)
        Cast(rewriteIR(v, in, memo), t)
      case NA(typ) =>
        assert(isSupertype(dep, typ))
        NA(dep)
      case IsNA(value) =>
        assert(dep.isOfType(TBoolean()))
        IsNA(rewriteIR(value, in, memo))
      case If(cond, cnsq, alt) =>
        val cond2 = rewriteIR(cond, in, memo)
        val cnsq2 = rewriteIR(cnsq, in, memo)
        val alt2 = rewriteIR(alt, in, memo)
        if (cnsq2.typ != alt2.typ)
          If(cond2, upcast(cnsq2, dep), upcast(alt2, dep))
        else
          If(cond2, cnsq2, alt2)
      case x@Let(name, value, body) =>
        val value2 = rewriteIR(value, in, memo)
        Let(
          name,
          value2,
          rewriteIR(body, in.bind(name, value2.typ), memo)
        )
      case Ref(name, t) =>
        assert(isSupertype(dep, t), s"ref dep is not a supertype of base:\n  ref to '$name'\n  dep:  ${dep.parsableString}\n  base: ${t.parsableString}")
        val envT = in.lookup(name)
        assert(isSupertype(dep, envT), s"ref dep is not a subtype of env type:\n  ref to '$name'\n  dep: ${dep.parsableString()}n  env: ${envT.parsableString()}")
        Ref(name, envT)
      case ApplyBinaryPrimOp(op, l, r) =>
        ApplyBinaryPrimOp(op, rewriteIR(l, in, memo), rewriteIR(r, in, memo))
      case ApplyUnaryPrimOp(op, x) =>
        ApplyUnaryPrimOp(op, rewriteIR(x, in, memo))
      case MakeArray(args, t) =>
        val depArray = dep.asInstanceOf[TArray]
        MakeArray(args.map(a => upcast(rewriteIR(a, in, memo), depArray.elementType)), dep.asInstanceOf[TArray])
      case ArrayRef(a, i) =>
        ArrayRef(rewriteIR(a, in, memo), rewriteIR(i, in, memo))
      case ArrayLen(a) =>
        ArrayLen(rewriteIR(a, in, memo))
      case ArrayRange(start, stop, step) =>
        ArrayRange(
          rewriteIR(start, in, memo),
          rewriteIR(stop, in, memo),
          rewriteIR(step, in, memo)
        )
      case ArrayMap(a, name, body) =>
        val a2 = rewriteIR(a, in, memo)
        ArrayMap(a2, name, rewriteIR(body, in.bind(name, -a2.typ.asInstanceOf[TArray].elementType), memo))
      case ArrayFilter(a, name, cond) =>
        val a2 = rewriteIR(a, in, memo)
        ArrayFilter(a2, name, rewriteIR(cond, in.bind(name, -a2.typ.asInstanceOf[TArray].elementType), memo))
      case ArrayFlatMap(a, name, body) =>
        val a2 = rewriteIR(a, in, memo)
        ArrayFlatMap(a2, name, rewriteIR(body, in.bind(name, -a2.typ.asInstanceOf[TArray].elementType), memo))
      case ArrayFold(a, zero, accumName, valueName, body) =>
        val a2 = rewriteIR(a, in, memo)
        val z2 = rewriteIR(zero, in, memo)
        ArrayFold(
          a2,
          z2,
          accumName,
          valueName,
          rewriteIR(body, in.bind(accumName -> z2.typ, valueName -> -a2.typ.asInstanceOf[TArray].elementType), memo)
        )
      case ArrayFor(a, valueName, body) =>
        val a2 = rewriteIR(a, in, memo)
        val body2 = rewriteIR(body, in.bind(valueName -> -a2.typ.asInstanceOf[TArray].elementType), memo)
        ArrayFor(a2, valueName, body2)
      case ApplyAggOp(a, constructorArgs, initOpArgs, signature) =>
        val a2 = rewriteIR(a, in, memo)
        ApplyAggOp(a2,
          constructorArgs.map(rewriteIR(_, in, memo)),
          initOpArgs.map(args => args.map(rewriteIR(_, in, memo))),
          signature)
      case SeqOp(a, i, agg) => SeqOp(rewriteIR(a, in, memo), rewriteIR(i, in, memo), agg)
      case Begin(xs) => Begin(xs.map(rewriteIR(_, in, memo)))
      case MakeStruct(fields) =>
        val depStruct = dep.asInstanceOf[TStruct]
        // drop unnecessary field IRs
        val depFields = depStruct.fieldNames.toSet
        MakeStruct(fields.flatMap { case (f, fir) =>
          if (depFields.contains(f))
            Some(f -> rewriteIR(fir, in, memo))
          else
            None
        })
      case InsertFields(old, fields) =>
        val depStruct = dep.asInstanceOf[TStruct]
        val depFields = depStruct.fieldNames.toSet
        InsertFields(rewriteIR(old, in, memo),
          fields.flatMap { case (f, fir) =>
            if (depFields.contains(f))
              Some(f -> rewriteIR(fir, in, memo))
            else
              None
          })
      case x@GetField(o, name) =>
        GetField(rewriteIR(o, in, memo), name)
      case x@MakeTuple(types) =>
        MakeTuple(types.map(rewriteIR(_, in, memo)))
      case GetTupleElement(o, idx) =>
        GetTupleElement(rewriteIR(o, in, memo), idx)
      case In(i, _) => ir
      case Die(_) => ir
      case Apply(function, args) =>
        Apply(function, args.map(rewriteIR(_, in, memo)))
      case ApplySpecial(function, args) =>
        ApplySpecial(function, args.map(rewriteIR(_, in, memo)))
    }
  }

  def upcast(ir: IR, rType: Type): IR = {
    if (ir.typ == rType)
      ir
    else {
      assert(isSupertype(rType, ir.typ), s"cannot upcast $ir:\n  sub:  ${ ir.typ.parsableString() }\n  super:  ${ rType.parsableString() }")
      ir.typ match {
        case ts: TStruct =>
          val rs = rType.asInstanceOf[TStruct]
          val uid = genUID()
          val ref = Ref(uid, ir.typ)
          val ms = MakeStruct(
            rs.fields.map { f =>
              f.name -> upcast(GetField(ref, f.name), f.typ)
            }
          )
          Let(uid, ir, ms)
        case ta: TArray =>
          val ra = rType.asInstanceOf[TArray]
          val uid = genUID()
          val ref = Ref(uid, ta.elementType)
          ArrayMap(ir, uid, upcast(ref, ra.elementType))
      }
    }
  }

  def upcast(mir: MatrixIR, mt: MatrixType): MatrixIR = {
    assert(isSupertype(mt, mir.typ))
    var ir = mir
    if (ir.typ.globalType != mt.globalType)
      ir = MatrixMapGlobals(ir, upcast(Ref("global", ir.typ.globalType), mt.globalType),
        BroadcastRow(Row(), TStruct(), HailContext.get.sc))
    if (ir.typ.colType != mt.colType)
      ir = MatrixMapCols(ir, upcast(Ref("sa", ir.typ.colType), mt.colType), None)
    if (ir.typ.rvRowType != mt.rvRowType)
      ir = MatrixMapRows(ir, upcast(Ref("va", ir.typ.rvRowType), mt.rvRowType), None)
    ir
  }

  def upcast(tir: TableIR, tt: TableType): TableIR = {
    assert(isSupertype(tt, tir.typ), s"cannot upcast \n  ir: $tir:\n  base: ${ tir.typ }\n  sub:  $tt")
    var ir = tir
    if (ir.typ.globalType != tt.globalType)
      ir = TableMapGlobals(ir, upcast(Ref("global", ir.typ.globalType), tt.globalType),
        BroadcastRow(Row(), TStruct(), HailContext.get.sc))
    if (ir.typ.rowType != tt.rowType)
      ir = TableMapRows(ir, upcast(Ref("row", ir.typ.rowType), tt.rowType), None)
    ir
  }
}

