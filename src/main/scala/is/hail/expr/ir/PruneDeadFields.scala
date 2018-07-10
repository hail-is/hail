package is.hail.expr.ir

import is.hail.annotations.AnnotationPathException
import is.hail.expr._
import is.hail.expr.types._
import is.hail.utils._

import scala.util.{Failure, Success, Try}

object PruneDeadFields {
  def subsetType(t: Type, path: Array[String], index: Int = 0): Type = {
    if (index == path.length)
      PruneDeadFields.minimal(t)
    else
      t match {
        case ts: TStruct => TStruct(ts.required, path(index) -> subsetType(ts.field(path(index)).typ, path, index + 1))
        case ta: TArray => TArray(subsetType(ta.elementType, path, index), ta.required)
      }
  }

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
        case (t1: TTuple, t2: TTuple) =>
          t1.required == t2.required &&
            t1.size == t2.size &&
            t1.types.zip(t2.types)
              .forall { case (elt1, elt2) => isSupertype(elt1, elt2) }
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
      val irCopy = ir.deepCopy()
      val memo = Memo.empty[BaseType]
      irCopy match {
        case mir: MatrixIR =>
          memoizeMatrixIR(mir, mir.typ, memo)
          rebuild(mir, memo)
        case tir: TableIR =>
          memoizeTableIR(tir, tir.typ, memo)
          rebuild(tir, memo)
        case vir: IR =>
          memoizeValueIR(vir, vir.typ, memo)
          rebuild(vir, Env.empty[Type], memo)
      }
    } catch {
      case e: Throwable => fatal(s"error trying to rebuild IR:\n${ Pretty(ir) }", e)
    }
  }


  def minimal(tt: TableType): TableType = {
    val keySet = tt.key.iterator.flatten.toSet
    tt.copy(
      rowType = tt.rowType.filterSet(keySet)._1,
      globalType = TStruct(tt.globalType.required)
    )
  }

  def minimal(mt: MatrixType): MatrixType = {
    val rowKeySet = mt.rowKey.toSet
    val colKeySet = mt.colKey.toSet
    mt.copy(
      rvRowType = TStruct(mt.rvRowType.required, mt.rvRowType.fields.flatMap { f =>
        if (rowKeySet.contains(f.name))
          Some(f.name -> f.typ)
        else if (f.name == MatrixType.entriesIdentifier)
          Some(f.name -> minimal(f.typ))
        else
          None
      }: _*),
      colType = mt.colType.filterSet(colKeySet)._1,
      globalType = TStruct(mt.globalType.required)
    )
  }

  def minimal[T <: Type](base: T): T = {
    val result = base match {
      case ts: TStruct => TStruct(ts.required)
      case ta: TArray => TArray(minimal(ta.elementType), ta.required)
      case t => t
    }
    result.asInstanceOf[T]
  }

  def unifyBaseType(base: BaseType, children: BaseType*): BaseType = unifyBaseTypeSeq(base, children)

  def unifyBaseTypeSeq(base: BaseType, children: Seq[BaseType]): BaseType = {
    base match {
      case tt: TableType =>
        val ttChildren = children.map(_.asInstanceOf[TableType])
        tt.copy(
          rowType = unify(tt.rowType, ttChildren.map(_.rowType): _*),
          globalType = unify(tt.globalType, ttChildren.map(_.globalType): _*)
        )
      case mt: MatrixType =>
        val mtChildren = children.map(_.asInstanceOf[MatrixType])
        mt.copy(
          globalType = unifySeq(mt.globalType, mtChildren.map(_.globalType)),
          rvRowType = unifySeq(mt.rvRowType, mtChildren.map(_.rvRowType)),
          colType = unifySeq(mt.colType, mtChildren.map(_.colType))
        )
      case t: Type =>
        if (children.isEmpty)
          return minimal(t)
        t match {
          case ts: TStruct =>
            val subStructs = children.map(_.asInstanceOf[TStruct])
            val subFields = ts.fields.map { f =>
              f -> subStructs.flatMap(s => s.fieldOption(f.name))
            }
              .filter(_._2.nonEmpty)
              .map { case (f, ss) => f.name -> unifySeq(f.typ, ss.map(_.typ)) }
            TStruct(ts.required, subFields: _*)
          case tt: TTuple =>
            val subTuples = children.map(_.asInstanceOf[TTuple])
            TTuple(tt.required, tt.types.indices.map(i => unifySeq(tt.types(i), subTuples.map(_.types(i)))): _*)
          case ta: TArray =>
            TArray(unifySeq(ta.elementType, children.map(_.asInstanceOf[TArray].elementType)), ta.required)
          case _ =>
            assert(children.forall(_.asInstanceOf[Type].isOfType(t)))
            base
        }
    }
  }

  def unify[T <: BaseType](base: T, children: T*): T = unifyBaseTypeSeq(base, children).asInstanceOf[T]

  def unifySeq[T <: BaseType](base: T, children: Seq[T]): T = unifyBaseTypeSeq(base, children).asInstanceOf[T]

  def unifyEnvs(envs: Env[(Type, Type)]*): Env[(Type, Type)] = unifyEnvsSeq(envs)

  def unifyEnvsSeq(envs: Seq[Env[(Type, Type)]]): Env[(Type, Type)] = {
    val lc = envs.lengthCompare(1)
    if (lc < 0)
      Env.empty[(Type, Type)]
    else if (lc == 0)
      envs.head
    else {
      val allKeys = envs.flatMap(_.m.keys).toSet
      val bindings = allKeys.toArray.map { k =>
        val envMatches = envs.flatMap(_.lookupOption(k))
        assert(envMatches.nonEmpty)
        val base = envMatches.head._1
        k -> (base, unifySeq(base, envMatches.map(_._2)))
      }
      new Env[(Type, Type)].bind(bindings: _*)
    }
  }

  def relationalTypeToEnv(bt: BaseType): Env[Type] = {
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

  def memoizeTableIR(tir: TableIR, requestedType: TableType, memo: Memo[BaseType]) {
    memo.bind(tir, requestedType)
    tir match {
      case TableRead(_, _, _, _) =>
      case TableLiteral(_) =>
      case TableParallelize(_, _, _) =>
      case TableImport(paths, typ, readerOpts) =>
      case TableRange(_, _) =>
      case x@TableJoin(left, right, joinType) =>
        val leftDep = left.typ.copy(
          rowType = TStruct(left.typ.rowType.required, left.typ.rowType.fieldNames.flatMap(f =>
            requestedType.rowType.fieldOption(f).map(reqF => f -> reqF.typ)): _*),
          globalType = requestedType.globalType)
        memoizeTableIR(left, leftDep, memo)

        val rightKeyFields = right.typ.key.iterator.flatten.toSet
        val rightDep = right.typ.copy(
          rowType = TStruct(right.typ.rowType.required, right.typ.rowType.fields.flatMap { f =>
            if (rightKeyFields.contains(f.name))
              Some(f.name -> f.typ)
            else
              requestedType.rowType.fieldOption(x.rightFieldMapping(f.name)).map(reqF => f.name -> reqF.typ)
          }: _*),
          globalType = minimal(requestedType.globalType))
        memoizeTableIR(right, rightDep, memo)
      case TableExplode(child, field) =>
        val minChild = minimal(child.typ)
        val dep2 = unify(child.typ, requestedType.copy(rowType = requestedType.rowType.filter(_.name != field)._1),
          minChild.copy(rowType = unify(
            child.typ.rowType,
            minChild.rowType,
            TStruct(field -> requestedType.rowType
              .fieldOption(field)
              .map(f => child.typ.rowType.field(field).typ match {
                case _: TArray => TArray(f.typ)
                case _: TSet => TSet(f.typ)
              }).getOrElse(minimal(child.typ.rowType.field(field).typ))))))
        memoizeTableIR(child, dep2, memo)
      case TableFilter(child, pred) =>
        val irDep = memoizeAndGetDep(pred, pred.typ, child.typ, memo)
        memoizeTableIR(child, unify(child.typ, requestedType, irDep), memo)
      case TableKeyBy(child, keys, nPartitionKeys, sort) =>
        memoizeTableIR(child, child.typ.copy(
          rowType = unify(child.typ.rowType, minimal(child.typ).rowType, requestedType.rowType),
          globalType = requestedType.globalType), memo)
      case TableOrderBy(child, sortFields) =>
        memoizeTableIR(child, child.typ.copy(
          rowType = unify(child.typ.rowType,
            child.typ.rowType.filterSet(sortFields.map(_.field).toSet)._1,
            requestedType.rowType),
          globalType = requestedType.globalType), memo)
      case TableDistinct(child) =>
        memoizeTableIR(child, requestedType, memo)
      case TableMapRows(child, newRow, newKey, preservedKeyFields) =>
        val rowDep = memoizeAndGetDep(newRow, requestedType.rowType, child.typ, memo)
        memoizeTableIR(child, unify(child.typ, minimal(child.typ).copy(globalType = requestedType.globalType), rowDep), memo)
      case TableMapGlobals(child, newRow, value) =>
        val globalDep = memoizeAndGetDep(newRow, requestedType.globalType, child.typ, memo)
        // fixme push down into value
        memoizeTableIR(child, unify(child.typ, requestedType.copy(globalType = globalDep.globalType), globalDep), memo)
      case TableAggregateByKey(child, newRow) =>
        val aggDep = memoizeAndGetDep(newRow, requestedType.rowType, child.typ, memo)
        memoizeTableIR(child, child.typ.copy(rowType = unify(child.typ.rowType, aggDep.rowType),
          globalType = unify(child.typ.globalType, aggDep.globalType, requestedType.globalType)), memo)
      case MatrixColsTable(child) =>
        val minChild = minimal(child.typ)
        val mtDep = minChild.copy(
          globalType = requestedType.globalType,
          colType = requestedType.rowType)
        memoizeMatrixIR(child, mtDep, memo)
      case MatrixRowsTable(child) =>
        val minChild = minimal(child.typ)
        val mtDep = minChild.copy(
          globalType = requestedType.globalType,
          rvRowType = unify(child.typ.rvRowType, minChild.rvRowType, requestedType.rowType))
        memoizeMatrixIR(child, mtDep, memo)
      case MatrixEntriesTable(child) =>
        val mtDep = child.typ.copy(
          globalType = requestedType.globalType,
          colType = TStruct(child.typ.colType.required,
            child.typ.colType.fields.flatMap(f => requestedType.rowType.fieldOption(f.name).map(f2 => f.name -> f2.typ)): _*),
          rvRowType = TStruct(child.typ.rvRowType.required, child.typ.rvRowType.fields.flatMap { f =>
            if (f.name == MatrixType.entriesIdentifier) {
              Some(f.name -> TArray(TStruct(child.typ.entryType.required, child.typ.entryType.fields.flatMap { entryField =>
                requestedType.rowType.fieldOption(entryField.name).map(f2 => f2.name -> f2.typ)
              }: _*), f.typ.required))
            } else {
              requestedType.rowType.fieldOption(f.name).map(f2 => f.name -> f2.typ)
            }
          }: _*
          ))
        memoizeMatrixIR(child, mtDep, memo)
      case TableUnion(children) =>
        children.foreach(memoizeTableIR(_, requestedType, memo))
      case TableUnkey(child) =>
        child.typ.key match {
          case Some(k) =>
            val childKeyFields = k.toSet
            memoizeTableIR(child, unify(child.typ, requestedType.copy(key = Some(k),
              rowType = unify(child.typ.rowType,
                requestedType.rowType,
                child.typ.rowType.filter(f => childKeyFields.contains(f.name))._1))),
              memo)
          case None => memoizeTableIR(child, requestedType, memo)
        }
    }
  }

  def memoizeMatrixIR(mir: MatrixIR, requestedType: MatrixType, memo: Memo[BaseType]) {
    memo.bind(mir, requestedType)
    mir match {
      case MatrixFilterCols(child, pred) =>
        val irDep = memoizeAndGetDep(pred, pred.typ, child.typ, memo)
        memoizeMatrixIR(child, unify(child.typ, requestedType, irDep), memo)
      case MatrixFilterRows(child, pred) =>
        val irDep = memoizeAndGetDep(pred, pred.typ, child.typ, memo)
        memoizeMatrixIR(child, unify(child.typ, requestedType, irDep), memo)
      case MatrixFilterEntries(child, pred) =>
        val irDep = memoizeAndGetDep(pred, pred.typ, child.typ, memo)
        memoizeMatrixIR(child, unify(child.typ, requestedType, irDep), memo)
      case MatrixMapEntries(child, newEntries) =>
        val irDep = memoizeAndGetDep(newEntries, requestedType.entryType, child.typ, memo)
        val depMod = requestedType.copy(rvRowType = TStruct(requestedType.rvRowType.required, requestedType.rvRowType.fields.map { f =>
          if (f.name == MatrixType.entriesIdentifier)
            f.name -> f.typ.asInstanceOf[TArray].copy(elementType = irDep.entryType)
          else
            f.name -> f.typ
        }: _*))
        memoizeMatrixIR(child, unify(child.typ, depMod, irDep), memo)
      case MatrixMapRows(child, newRow, newKey) =>
        val irDep = memoizeAndGetDep(newRow, requestedType.rowType, child.typ, memo)
        val depMod = child.typ.copy(rvRowType = TStruct(irDep.rvRowType.fields.map { f =>
          if (f.name == MatrixType.entriesIdentifier)
            f.name -> unify(child.typ.rvRowType.field(MatrixType.entriesIdentifier).typ,
              f.typ,
              requestedType.rvRowType.field(MatrixType.entriesIdentifier).typ)
          else
            f.name -> f.typ
        }: _*), colType = requestedType.colType, globalType = requestedType.globalType)
        memoizeMatrixIR(child, unify(child.typ, depMod, irDep), memo)
      case MatrixMapCols(child, newCol, newKey) =>
        val irDep = memoizeAndGetDep(newCol, requestedType.colType, child.typ, memo)
        val depMod = minimal(child.typ).copy(rvRowType = requestedType.rvRowType,
          globalType = requestedType.globalType)
        memoizeMatrixIR(child, unify(child.typ, depMod, irDep), memo)
      case MatrixMapGlobals(child, newRow, value) =>
        val irDep = memoizeAndGetDep(newRow, requestedType.globalType, child.typ, memo)
        // FIXME push down into value
        memoizeMatrixIR(child, unify(child.typ, requestedType.copy(globalType = irDep.globalType), irDep), memo)
      case MatrixRead(_, _, _, _, _, _) =>
      case MatrixLiteral(typ, value) =>
      case MatrixChooseCols(child, oldIndices) =>
        memoizeMatrixIR(child, requestedType, memo)
      case MatrixCollectColsByKey(child) =>
        val colKeySet = requestedType.colKey.toSet
        val explodedDep = requestedType.copy(
          colType = TStruct(requestedType.colType.required, requestedType.colType.fields.map { f =>
            if (colKeySet.contains(f.name))
              f.name -> f.typ
            else {
              f.name -> f.typ.asInstanceOf[TArray].elementType
            }
          }: _*),
          rvRowType = requestedType.rvRowType.copy(fields = requestedType.rvRowType.fields.map { f =>
            if (f.name == MatrixType.entriesIdentifier)
              f.copy(typ = TArray(
                TStruct(requestedType.entryType.required, requestedType.entryType.fields.map(ef =>
                  ef.name -> ef.typ.asInstanceOf[TArray].elementType): _*), f.typ.required))
            else
              f
          })
        )
        memoizeMatrixIR(child, explodedDep, memo)
      case MatrixAggregateRowsByKey(child, expr) =>
        val irDep = memoizeAndGetDep(expr, requestedType.entryType, child.typ, memo)
        val childDep = child.typ.copy(
          globalType = unify(child.typ.globalType, irDep.globalType, requestedType.globalType),
          rvRowType = irDep.rvRowType,
          colType = unify(child.typ.colType, irDep.colType, requestedType.colType)
        )
        memoizeMatrixIR(child, childDep, memo)
      case MatrixAggregateColsByKey(child, expr) =>
        val irDep = memoizeAndGetDep(expr, requestedType.entryType, child.typ, memo)
        val rvRowDep = TStruct(
          child.typ.rvRowType.required, child.typ.rvRowType.fields.flatMap { f =>
            if (f.name == MatrixType.entriesIdentifier)
              Some(f.name -> irDep.entryArrayType)
            else {
              val requestedFieldDep = requestedType.rvRowType.fieldOption(f.name).map(_.typ)
              val irFieldDep = irDep.rvRowType.fieldOption(f.name).map(_.typ)
              if (requestedFieldDep.isEmpty && irFieldDep.isEmpty)
                None
              else
                Some(f.name -> unifySeq(f.typ, (requestedFieldDep.iterator ++ irFieldDep.iterator).toFastSeq))
            }
          }: _*
        )
        val childDep = child.typ.copy(
          globalType = unify(child.typ.globalType, irDep.globalType, requestedType.globalType),
          rvRowType = rvRowDep,
          colType = irDep.colType
        )
        memoizeMatrixIR(child, childDep, memo)
      case MatrixExplodeRows(child, path) =>
        val baseType = child.typ.rowType.queryTyped(path.toList)._1
        val fieldDep = Try(requestedType.rowType.queryTyped(path.toList)._1) match {
          case Success(t) => baseType match {
            case ta: TArray => ta.copy(elementType = t)
            case ts: TSet => ts.copy(elementType = t)
          }
          case Failure(_) => minimal(baseType)
        }
        val minChild = minimal(child.typ)
        val dep = requestedType.copy(rvRowType = unify(child.typ.rvRowType,
          requestedType.rvRowType.insert(fieldDep, path.toList)._1.asInstanceOf[TStruct]))
        memoizeMatrixIR(child, dep, memo)
    }
  }


  def memoizeAndGetDep(ir: IR, requestedType: Type, base: TableType, memo: Memo[BaseType]): TableType = {
    val depEnv = memoizeValueIR(ir, requestedType, memo).m.mapValues(_._2)
    val min = minimal(base)
    val rowArgs = (Iterator.single(min.rowType) ++
      depEnv.get("row").map(_.asInstanceOf[TStruct]).iterator).toArray
    val globalArgs = (Iterator.single(min.globalType) ++ depEnv.get("global").map(_.asInstanceOf[TStruct]).iterator).toArray
    base.copy(rowType = unifySeq(base.rowType, rowArgs),
      globalType = unifySeq(base.globalType, globalArgs))
  }

  def memoizeAndGetDep(ir: IR, requestedType: Type, base: MatrixType, memo: Memo[BaseType]): MatrixType = {
    val depEnv = memoizeValueIR(ir, requestedType, memo).m.mapValues(_._2)
    val min = minimal(base)
    val eField = base.rvRowType.field(MatrixType.entriesIdentifier)
    val rowArgs = (Iterator.single(min.rvRowType) ++ depEnv.get("va").iterator ++
      Iterator.single(TStruct(eField.name -> TArray(
        unifySeq(eField.typ.asInstanceOf[TArray].elementType,
          depEnv.get("g").iterator.toFastSeq),
        eField.typ.required)))).toFastSeq
    val colArgs = (Iterator.single(min.colType) ++ depEnv.get("sa").iterator).toFastSeq
    val globalArgs = (Iterator.single(min.globalType) ++ depEnv.get("global").iterator).toFastSeq
    base.copy(rvRowType = unifySeq(base.rvRowType, rowArgs).asInstanceOf[TStruct],
      globalType = unifySeq(base.globalType, globalArgs).asInstanceOf[TStruct],
      colType = unifySeq(base.colType, colArgs).asInstanceOf[TStruct])
  }

  /**
    * This function does *not* necessarily bind each child node in `memo`.
    * Known dead code is not memoized. For instance:
    *
    *   ir = MakeStruct(Seq("a" -> (child1), "b" -> (child2)))
    *   requestedType = TStruct("a" -> (reqType of a))
    *
    * In the above, `child2` will not be memoized because `ir` does not require
    * any of the "b" dependencies in order to create its own requested type,
    * which only contains "a".
    */
  def memoizeValueIR(ir: IR, requestedType: Type, memo: Memo[BaseType]): Env[(Type, Type)] = {
    memo.bind(ir, requestedType)
    ir match {
      case IsNA(value) => memoizeValueIR(value, minimal(value.typ), memo)
      case If(cond, cnsq, alt) =>
        unifyEnvs(
          memoizeValueIR(cond, cond.typ, memo),
          memoizeValueIR(cnsq, requestedType, memo),
          memoizeValueIR(alt, requestedType, memo)
        )
      case Let(name, value, body) =>
        val bodyEnv = memoizeValueIR(body, requestedType, memo)
        val valueType = bodyEnv.lookupOption(name).map(_._2).getOrElse(minimal(value.typ))
        unifyEnvs(
          bodyEnv.delete(name),
          memoizeValueIR(value, valueType, memo)
        )
      case Ref(name, t) =>
        Env.empty[(Type, Type)].bind(name, t -> requestedType)
      case MakeArray(args, _) =>
        val eltType = requestedType.asInstanceOf[TArray].elementType
        unifyEnvsSeq(args.map(a => memoizeValueIR(a, eltType, memo)))
      case ArrayRef(a, i) =>
        unifyEnvs(
          memoizeValueIR(a, a.typ.asInstanceOf[TArray].copy(elementType = requestedType), memo),
          memoizeValueIR(i, i.typ, memo))
      case ArrayLen(a) =>
        memoizeValueIR(a, minimal(a.typ), memo)
      case ArrayMap(a, name, body) =>
        val aType = a.typ.asInstanceOf[TArray]
        val bodyDep = memoizeValueIR(body,
          requestedType.asInstanceOf[TArray].elementType,
          memo)
        val valueType = bodyDep.lookupOption(name).map(_._2).getOrElse(minimal(-aType.elementType))
        unifyEnvs(
          bodyDep.delete(name),
          memoizeValueIR(a, aType.copy(elementType = valueType), memo)
        )
      case ArrayFilter(a, name, cond) =>
        val aType = a.typ.asInstanceOf[TArray]
        val bodyEnv = memoizeValueIR(cond, cond.typ, memo)
        val valueType = unify(aType.elementType,
          requestedType.asInstanceOf[TArray].elementType,
          bodyEnv.lookupOption(name).map(_._2).getOrElse(minimal(-aType.elementType)))
        unifyEnvs(
          bodyEnv.delete(name),
          memoizeValueIR(a, aType.copy(elementType = valueType), memo)
        )
      case ArrayFlatMap(a, name, body) =>
        val aType = a.typ.asInstanceOf[TArray]
        val bodyEnv = memoizeValueIR(body, requestedType, memo)
        val valueType = bodyEnv.lookupOption(name).map(_._2).getOrElse(minimal(-aType.elementType))
        unifyEnvs(
          bodyEnv.delete(name),
          memoizeValueIR(a, aType.copy(elementType = valueType), memo)
        )
      case ArrayFold(a, zero, accumName, valueName, body) =>
        val aType = a.typ.asInstanceOf[TArray]
        val zeroEnv = memoizeValueIR(zero, zero.typ, memo)
        val bodyEnv = memoizeValueIR(body, body.typ, memo)
        val valueType = bodyEnv.lookupOption(valueName).map(_._2).getOrElse(minimal(-aType.elementType))
        unifyEnvs(
          zeroEnv,
          bodyEnv.delete(accumName).delete(valueName),
          memoizeValueIR(a, aType.copy(elementType = valueType), memo)
        )
      case ArrayFor(a, valueName, body) =>
        assert(requestedType == TVoid)
        val aType = a.typ.asInstanceOf[TArray]
        val bodyEnv = memoizeValueIR(body, body.typ, memo)
        val valueType = bodyEnv.lookupOption(valueName).map(_._2).getOrElse(minimal(-aType.elementType))
        unifyEnvs(
          bodyEnv.delete(valueName),
          memoizeValueIR(a, aType.copy(elementType = valueType), memo)
        )
      case MakeStruct(fields) =>
        val sType = requestedType.asInstanceOf[TStruct]
        unifyEnvsSeq(fields.flatMap { case (fname, fir) =>
          // ignore unreachable fields, these are eliminated on the upwards pass
          sType.fieldOption(fname).map(f => memoizeValueIR(fir, f.typ, memo))
        })
      case InsertFields(old, fields) =>
        val sType = requestedType.asInstanceOf[TStruct]
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
        unifyEnvsSeq(
          FastSeq(memoizeValueIR(old, leftDep, memo)) ++
            // ignore unreachable fields, these are eliminated on the upwards pass
            fields.flatMap { case (fname, fir) =>
              rightDep.fieldOption(fname).map(f => memoizeValueIR(fir, f.typ, memo))
            }
        )
      case SelectFields(old, fields) =>
        val sType = requestedType.asInstanceOf[TStruct]
        memoizeValueIR(old, TStruct(old.typ.required, fields.flatMap(f => sType.fieldOption(f).map(f -> _.typ)): _*), memo)
      case GetField(o, name) =>
        memoizeValueIR(o, TStruct(o.typ.required, name -> requestedType), memo)
      case MakeTuple(types) =>
        val tType = requestedType.asInstanceOf[TTuple]
        assert(types.length == tType.size)
        unifyEnvsSeq(
          types.zip(tType.types).map { case (tir, t) => memoizeValueIR(tir, t, memo) }
        )
      case GetTupleElement(o, idx) =>
        // FIXME handle tuples better
        val childTupleType = o.typ.asInstanceOf[TTuple]
        val tupleDep = TTuple(childTupleType.required,
          childTupleType.types
            .zipWithIndex
            .map { case (t, i) => if (i == idx) requestedType else minimal(t) }: _*)
        memoizeValueIR(o, tupleDep, memo)
      case TableCount(child) =>
        memoizeTableIR(child, minimal(child.typ), memo)
        Env.empty[(Type, Type)]
      case TableAggregate(child, query) =>
        val queryDep = memoizeAndGetDep(query, query.typ, child.typ, memo)
        memoizeTableIR(child, queryDep, memo)
        Env.empty[(Type, Type)]
      case _: IR =>
        val envs = ir.children.flatMap {
          case mir: MatrixIR =>
            memoizeMatrixIR(mir, mir.typ, memo)
            None
          case tir: TableIR =>
            memoizeTableIR(tir, tir.typ, memo)
            None
          case ir: IR =>
            Some(memoizeValueIR(ir, ir.typ, memo))
        }
        unifyEnvsSeq(envs)
    }
  }

  def rebuild(tir: TableIR, memo: Memo[BaseType]): TableIR = {
    val dep = memo.lookup(tir).asInstanceOf[TableType]
    tir match {
      case TableImport(paths, typ, readerOpts) =>
        val fieldsToRead = readerOpts.originalType.fields.flatMap(f => dep.rowType.fieldOption(f.name).map(_ => f.index)).toArray
        val newTyp = typ.copy(rowType = TStruct(typ.rowType.required,
          fieldsToRead.map(i => readerOpts.originalType.fieldNames(i) -> readerOpts.originalType.types(i)): _*))
        TableImport(paths, newTyp, readerOpts.copy(useColIndices = fieldsToRead))
      case TableRead(path, spec, _, dropRows) => TableRead(path, spec, dep, dropRows)
      case TableFilter(child, pred) =>
        val child2 = rebuild(child, memo)
        val pred2 = rebuild(pred, child2.typ, memo)
        TableFilter(child2, pred2)
      case TableMapRows(child, newRow, newKey, preservedKeyFields) =>
        val child2 = rebuild(child, memo)
        val newRow2 = rebuild(newRow, child2.typ, memo)
        TableMapRows(child2, newRow2, newKey, preservedKeyFields)
      case TableMapGlobals(child, newRow, value) =>
        // fixme push down into value
        val child2 = rebuild(child, memo)
        TableMapGlobals(child2, rebuild(newRow, child2.typ, memo, "value" -> value.t), value)
      case TableAggregateByKey(child, expr) =>
        val child2 = rebuild(child, memo)
        TableAggregateByKey(child2, rebuild(expr, child2.typ, memo))
      case _ => tir.copy(tir.children.map {
        // IR should be a match error - all nodes with child value IRs should have a rule
        case childT: TableIR => rebuild(childT, memo)
        case childM: MatrixIR => rebuild(childM, memo)
      }).asInstanceOf[TableIR]
    }
  }

  def rebuild(mir: MatrixIR, memo: Memo[BaseType]): MatrixIR = {
    val dep = memo.lookup(mir).asInstanceOf[MatrixType]
    mir match {
      case x@MatrixRead(typ, partitionCounts, colCount, dropCols, dropRows, reader) =>
        MatrixRead(dep, partitionCounts, colCount, dropCols, dropRows, reader)
      case MatrixFilterCols(child, pred) =>
        val child2 = rebuild(child, memo)
        MatrixFilterCols(child2, rebuild(pred, child2.typ, memo))
      case MatrixFilterRows(child, pred) =>
        val child2 = rebuild(child, memo)
        MatrixFilterRows(child2, rebuild(pred, child2.typ, memo))
      case MatrixFilterEntries(child, pred) =>
        val child2 = rebuild(child, memo)
        MatrixFilterEntries(child2, rebuild(pred, child2.typ, memo))
      case MatrixMapEntries(child, newEntries) =>
        val child2 = rebuild(child, memo)
        MatrixMapEntries(child2, rebuild(newEntries, child2.typ, memo))
      case MatrixMapRows(child, newRow, newKey) =>
        val child2 = rebuild(child, memo)
        MatrixMapRows(child2, rebuild(newRow, child2.typ, memo), newKey)
      case MatrixMapCols(child, newCol, newKey) =>
        // FIXME account for key
        val child2 = rebuild(child, memo)
        MatrixMapCols(child2, rebuild(newCol, child2.typ, memo), newKey)
      case MatrixMapGlobals(child, newRow, value) =>
        val child2 = rebuild(child, memo)
        MatrixMapGlobals(child2, rebuild(newRow, child2.typ, memo, "value" -> value.t), value)
      case MatrixAggregateRowsByKey(child, expr) =>
        val child2 = rebuild(child, memo)
        MatrixAggregateRowsByKey(child2, rebuild(expr, child2.typ, memo))
      case MatrixAggregateColsByKey(child, expr) =>
        val child2 = rebuild(child, memo)
        MatrixAggregateColsByKey(child2, rebuild(expr, child2.typ, memo))
      case _ => mir.copy(mir.children.map {
        // IR should be a match error - all nodes with child value IRs should have a rule
        case childT: TableIR => rebuild(childT, memo)
        case childM: MatrixIR => rebuild(childM, memo)
      }).asInstanceOf[MatrixIR]

    }
  }

  def rebuild(ir: IR, in: BaseType, memo: Memo[BaseType], bindings: (String, Type)*): IR = {
    rebuild(ir, relationalTypeToEnv(in).bind(bindings: _*), memo)
  }

  def rebuild(ir: IR, in: Env[Type], memo: Memo[BaseType]): IR = {
    val requestedType = memo.lookup(ir).asInstanceOf[Type]
    ir match {
      case NA(typ) => NA(requestedType)
      case If(cond, cnsq, alt) =>
        val cond2 = rebuild(cond, in, memo)
        val cnsq2 = rebuild(cnsq, in, memo)
        val alt2 = rebuild(alt, in, memo)
        if (cnsq2.typ != alt2.typ)
          If(cond2, upcast(cnsq2, requestedType), upcast(alt2, requestedType))
        else
          If(cond2, cnsq2, alt2)
      case Let(name, value, body) =>
        val value2 = rebuild(value, in, memo)
        Let(
          name,
          value2,
          rebuild(body, in.bind(name, value2.typ), memo)
        )
      case Ref(name, t) =>
        Ref(name, in.lookupOption(name).getOrElse(t))
      case MakeArray(args, t) =>
        val depArray = requestedType.asInstanceOf[TArray]
        MakeArray(args.map(a => upcast(rebuild(a, in, memo), depArray.elementType)), requestedType.asInstanceOf[TArray])
      case ArrayMap(a, name, body) =>
        val a2 = rebuild(a, in, memo)
        ArrayMap(a2, name, rebuild(body, in.bind(name, -a2.typ.asInstanceOf[TArray].elementType), memo))
      case ArrayFilter(a, name, cond) =>
        val a2 = rebuild(a, in, memo)
        ArrayFilter(a2, name, rebuild(cond, in.bind(name, -a2.typ.asInstanceOf[TArray].elementType), memo))
      case ArrayFlatMap(a, name, body) =>
        val a2 = rebuild(a, in, memo)
        ArrayFlatMap(a2, name, rebuild(body, in.bind(name, -a2.typ.asInstanceOf[TArray].elementType), memo))
      case ArrayFold(a, zero, accumName, valueName, body) =>
        val a2 = rebuild(a, in, memo)
        val z2 = rebuild(zero, in, memo)
        ArrayFold(
          a2,
          z2,
          accumName,
          valueName,
          rebuild(body, in.bind(accumName -> z2.typ, valueName -> -a2.typ.asInstanceOf[TArray].elementType), memo)
        )
      case ArrayFor(a, valueName, body) =>
        val a2 = rebuild(a, in, memo)
        val body2 = rebuild(body, in.bind(valueName -> -a2.typ.asInstanceOf[TArray].elementType), memo)
        ArrayFor(a2, valueName, body2)
      case MakeStruct(fields) =>
        val depStruct = requestedType.asInstanceOf[TStruct]
        // drop unnecessary field IRs
        val depFields = depStruct.fieldNames.toSet
        MakeStruct(fields.flatMap { case (f, fir) =>
          if (depFields.contains(f))
            Some(f -> rebuild(fir, in, memo))
          else {
            log.info(s"Prune: MakeStruct: eliminating field '$f'")
            None
          }
        })
      case InsertFields(old, fields) =>
        val depStruct = requestedType.asInstanceOf[TStruct]
        val depFields = depStruct.fieldNames.toSet
        InsertFields(rebuild(old, in, memo),
          fields.flatMap { case (f, fir) =>
            if (depFields.contains(f))
              Some(f -> rebuild(fir, in, memo))
            else {
              log.info(s"Prune: InsertFields: eliminating field '$f'")
              None
            }
          })
      case SelectFields(old, fields) =>
        val depStruct = requestedType.asInstanceOf[TStruct]
        val old2 = rebuild(old, in, memo)
        SelectFields(old2, fields.filter(f => old2.typ.asInstanceOf[TStruct].hasField(f) && depStruct.hasField(f)))
      case Uniroot(argname, function, min, max) =>
        assert(requestedType == TFloat64Optional)
        Uniroot(argname,
          rebuild(function, in.bind(argname -> TFloat64Optional), memo),
          rebuild(min, in, memo),
          rebuild(max, in, memo))
      case TableAggregate(child, query) =>
        val child2 = rebuild(child, memo)
        val query2 = rebuild(query, child2.typ, memo)
        TableAggregate(child2, query2)
      case _ =>
        ir.copy(ir.children.map {
          case valueIR: IR => rebuild(valueIR, in, memo)
          case mir: MatrixIR => rebuild(mir, memo)
          case tir: TableIR => rebuild(tir, memo)
        }).asInstanceOf[IR]
    }
  }

  def upcast(ir: IR, rType: Type): IR = {
    if (ir.typ == rType)
      ir
    else {
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
}

