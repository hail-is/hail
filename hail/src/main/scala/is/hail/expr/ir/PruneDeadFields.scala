package is.hail.expr.ir

import is.hail.HailContext
import is.hail.annotations._
import is.hail.expr.types.{virtual, _}
import is.hail.expr.types.virtual._
import is.hail.utils._

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
        case (TArray(et1, r1), TArray(et2, r2)) => (!r1 || r2) && isSupertype(et1, et2)
        case (TSet(et1, r1), TSet(et2, r2)) => (!r1 || r2) && isSupertype(et1, et2)
        case (TDict(kt1, vt1, r1), TDict(kt2, vt2, r2)) => (!r1 || r2) && isSupertype(kt1, kt2) && isSupertype(vt1, vt2)
        case (s1: TStruct, s2: TStruct) =>
          var idx = -1
          (!s1.required || s2.required) && s1.fields.forall { f =>
            val s2field = s2.field(f.name)
            if (s2field.index > idx) {
              idx = s2field.index
              isSupertype(f.typ, s2field.typ)
            } else
              false
          }
        case (t1: TTuple, t2: TTuple) =>
          (!t1.required || t2.required) &&
            t1.size == t2.size &&
            t1.types.zip(t2.types)
              .forall { case (elt1, elt2) => isSupertype(elt1, elt2) }
        case (t1: Type, t2: Type) => t1 == t2 || t2.required && t2.isOfType(t1)
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
        case bmir: BlockMatrixIR => bmir //NOTE Currently no BlockMatrixIRs would have dead fields
        case vir: IR =>
          memoizeValueIR(vir, vir.typ, memo)
          rebuild(vir, Env.empty[Type], memo)
      }
    } catch {
      case e: Throwable => fatal(s"error trying to rebuild IR:\n${ Pretty(ir) }", e)
    }
  }

  def pruneColValues(mv: MatrixValue, valueIR: IR, isArray: Boolean = false): (Type, BroadcastIndexedSeq, IR) = {
    val matrixType = mv.typ
    val oldColValues = mv.colValues
    val oldColType = matrixType.colType
    val memo = Memo.empty[BaseType]
    val valueIRCopy = valueIR.deepCopy()
    val colDep = memoizeValueIR(valueIRCopy, valueIR.typ, memo)
      .m.mapValues(_._2)
      .getOrElse("sa", if (isArray) TArray(TStruct()) else TStruct())
    if (colDep != oldColType)
      log.info(s"pruned col values:\n  From: $oldColType\n  To: ${ colDep }")
    val newColsType = if (isArray) colDep.asInstanceOf[TArray] else TArray(colDep)
    val newIndexedSeq = Interpret[IndexedSeq[Annotation]](
      upcast(Ref("values", TArray(oldColType)), newColsType),
      Env.empty[(Any, Type)]
        .bind("values" -> (mv.colValues.value, TArray(oldColType))),
      FastIndexedSeq(),
      None,
      optimize = false)
    (colDep,
      BroadcastIndexedSeq(newIndexedSeq, newColsType, mv.sparkContext),
      rebuild(valueIRCopy, relationalTypeToEnv(matrixType).bind("sa" -> colDep), memo)
    )
  }

  def minimal(tt: TableType): TableType = {
    val keySet = tt.key.toSet
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

  def minimalBT[T <: BaseType](base: T): T = {
    (base match {
      case tt: TableType => minimal(tt)
      case mt: MatrixType => minimal(mt)
      case t: Type => minimal(t)
    }).asInstanceOf[T]
  }

  def unifyBaseType(base: BaseType, children: BaseType*): BaseType = unifyBaseTypeSeq(base, children)

  def unifyBaseTypeSeq(base: BaseType, children: Seq[BaseType]): BaseType = {
    if (children.isEmpty)
      return minimalBT(base)
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
      case mt: MatrixType =>
        Env.empty[Type]
          .bind("global", mt.globalType)
          .bind("sa", mt.colType)
          .bind("va", mt.rvRowType)
          .bind("g", mt.entryType)
    }
  }

  def memoizeTableIR(tir: TableIR, requestedType: TableType, memo: Memo[BaseType]) {
    memo.bind(tir, requestedType)
    tir match {
      case TableRead(_, _, _) =>
      case TableLiteral(_) =>
      case TableParallelize(rowsAndGlobal, _) =>
        memoizeValueIR(rowsAndGlobal, TStruct("rows" -> TArray(requestedType.rowType), "global" -> requestedType.globalType), memo)
      case TableRange(_, _) =>
      case TableRepartition(child, _, _) => memoizeTableIR(child, requestedType, memo)
      case TableHead(child, _) => memoizeTableIR(child, requestedType, memo)
      case x@TableJoin(left, right, _, _) =>
        val leftDep = left.typ.copy(
          rowType = TStruct(left.typ.rowType.required, left.typ.rowType.fieldNames.flatMap(f =>
            requestedType.rowType.fieldOption(f).map(reqF => f -> reqF.typ)): _*),
          globalType = TStruct(left.typ.globalType.required, left.typ.globalType.fieldNames.flatMap(f =>
            requestedType.globalType.fieldOption(f).map(reqF => f -> reqF.typ)): _*))
        memoizeTableIR(left, leftDep, memo)
        val rightKeyFields = right.typ.key.toSet
        val rightDep = right.typ.copy(
          rowType = TStruct(right.typ.rowType.required, right.typ.rowType.fieldNames.flatMap(f =>
            if (rightKeyFields.contains(f))
              Some(f -> right.typ.rowType.field(f).typ)
            else
              requestedType.rowType.fieldOption(f).map(reqF => f -> reqF.typ)): _*),
          globalType = TStruct(right.typ.globalType.required, right.typ.globalType.fieldNames.flatMap(f =>
            requestedType.globalType.fieldOption(f).map(reqF => f -> reqF.typ)): _*))
        memoizeTableIR(right, rightDep, memo)
      case TableLeftJoinRightDistinct(left, right, root) =>
        val fieldDep = requestedType.rowType.fieldOption(root).map(_.typ.asInstanceOf[TStruct])
        fieldDep match {
          case Some(struct) =>
            val rightDep = right.typ.copy(rowType = unify(
              right.typ.rowType,
              FastIndexedSeq[TStruct](right.typ.rowType.filterSet(right.typ.key.toSet, true)._1) ++
                FastIndexedSeq(struct): _*),
              globalType = minimal(right.typ.globalType))
            memoizeTableIR(right, rightDep, memo)
            val leftDep = unify(
              left.typ,
              requestedType.copy(rowType =
                requestedType.rowType.filterSet(Set(root), include = false)._1))
            memoizeTableIR(left, leftDep, memo)
          case None =>
            // don't memoize right if we are going to elide it during rebuild
            memoizeTableIR(left, requestedType, memo)
        }
      case TableIntervalJoin(left, right, root) =>
        val fieldDep = requestedType.rowType.fieldOption(root).map(_.typ.asInstanceOf[TStruct])
        fieldDep match {
          case Some(struct) =>
            val rightDep = right.typ.copy(rowType = unify(
              right.typ.rowType,
              FastIndexedSeq[TStruct](right.typ.rowType.filterSet(right.typ.key.toSet, true)._1) ++
                FastIndexedSeq(struct): _*),
              globalType = minimal(right.typ.globalType))
            memoizeTableIR(right, rightDep, memo)
            val leftDep = unify(
              left.typ,
              requestedType.copy(rowType =
                requestedType.rowType.filterSet(Set(root), include = false)._1))
            memoizeTableIR(left, leftDep, memo)
          case None =>
            // don't memoize right if we are going to elide it during rebuild
            memoizeTableIR(left, requestedType, memo)
        }
      case TableMultiWayZipJoin(children, fieldName, globalName) =>
        val gType = requestedType.globalType.fieldOption(globalName)
          .map(_.typ.asInstanceOf[TArray].elementType)
          .getOrElse(TStruct()).asInstanceOf[TStruct]
        val rType = requestedType.rowType.fieldOption(fieldName)
          .map(_.typ.asInstanceOf[TArray].elementType)
          .getOrElse(TStruct()).asInstanceOf[TStruct]
        val child1 = children.head
        val dep = child1.typ.copy(
          rowType = TStruct(child1.typ.rowType.required, child1.typ.rowType.fieldNames.flatMap(f =>
              child1.typ.keyType.fieldOption(f).orElse(rType.fieldOption(f)).map(reqF => f -> reqF.typ)
            ): _*),
          globalType = gType)
        children.foreach(memoizeTableIR(_, dep, memo))
      case TableExplode(child, path) =>
        def getExplodedField(typ: TableType): Type = typ.rowType.queryTyped(path.toList)._1
        val preExplosionFieldType = getExplodedField(child.typ)
        val prunedPreExlosionFieldType = try {
          val t = getExplodedField(requestedType)
          preExplosionFieldType match {
            case ta: TArray => ta.copy(elementType = t)
            case ts: TSet => ts.copy(elementType = t)
          }
        } catch {
          case e: AnnotationPathException => minimal(preExplosionFieldType)
        }
        val dep = requestedType.copy(rowType = unify(child.typ.rowType,
          requestedType.rowType.insert(prunedPreExlosionFieldType, path.toList)._1.asInstanceOf[TStruct]))
        memoizeTableIR(child, dep, memo)
      case TableFilter(child, pred) =>
        val irDep = memoizeAndGetDep(pred, pred.typ, child.typ, memo)
        memoizeTableIR(child, unify(child.typ, requestedType, irDep), memo)
      case TableKeyBy(child, keys, isSorted) =>
        memoizeTableIR(child, child.typ.copy(
          rowType = unify(child.typ.rowType, minimal(child.typ).rowType, requestedType.rowType),
          globalType = requestedType.globalType), memo)
      case TableOrderBy(child, sortFields) =>
        memoizeTableIR(child, child.typ.copy(
          rowType = unify(child.typ.rowType,
            child.typ.rowType.filterSet((sortFields.map(_.field) ++ child.typ.key).toSet)._1,
            requestedType.rowType),
          globalType = requestedType.globalType), memo)
      case TableDistinct(child) =>
        memoizeTableIR(child, requestedType, memo)
      case TableMapRows(child, newRow) =>
        val rowDep = memoizeAndGetDep(newRow, requestedType.rowType, child.typ, memo)
        memoizeTableIR(child, unify(child.typ, minimal(child.typ).copy(globalType = requestedType.globalType), rowDep), memo)
      case TableMapGlobals(child, newGlobals) =>
        val globalDep = memoizeAndGetDep(newGlobals, requestedType.globalType, child.typ, memo)
        memoizeTableIR(child, unify(child.typ, requestedType.copy(globalType = globalDep.globalType), globalDep), memo)
      case TableAggregateByKey(child, newRow) =>
        val aggDep = memoizeAndGetDep(newRow, requestedType.rowType, child.typ, memo)
        memoizeTableIR(child, child.typ.copy(rowType = unify(child.typ.rowType, aggDep.rowType),
          globalType = unify(child.typ.globalType, aggDep.globalType, requestedType.globalType)), memo)
      case TableKeyByAndAggregate(child, expr, newKey, _, _) =>
        val keyDep = memoizeAndGetDep(newKey, newKey.typ, child.typ, memo)
        val exprDep = memoizeAndGetDep(expr, requestedType.valueType, child.typ, memo)
        memoizeTableIR(child,
          unify(child.typ, keyDep, exprDep, minimal(child.typ).copy(globalType = requestedType.globalType)),
          memo)
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
      case CastMatrixToTable(child, entriesFieldName, colsFieldName) =>
        val minChild = minimal(child.typ)
        val m = Map(entriesFieldName -> MatrixType.entriesIdentifier)
        val childDep = minChild.copy(
          globalType = if (requestedType.globalType.hasField(colsFieldName))
              requestedType.globalType.deleteKey(colsFieldName)
            else
              requestedType.globalType,
          colType = if (requestedType.globalType.hasField(colsFieldName))
            unify(child.typ.colType, minChild.colType,
              requestedType.globalType.field(colsFieldName).typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct])
            else
              minChild.colType,
          rvRowType = unify(child.typ.rvRowType, minChild.rvRowType, requestedType.rowType.rename(m)))
        memoizeMatrixIR(child, childDep, memo)
      case TableRename(child, rowMap, globalMap) =>
        val rowMapRev = rowMap.map { case (k, v) => (v, k) }
        val globalMapRev = globalMap.map { case (k, v) => (v, k) }
        val childDep = TableType(
          rowType = requestedType.rowType.rename(rowMapRev),
          globalType = requestedType.globalType.rename(globalMapRev),
          key = requestedType.key.map(k => rowMapRev.getOrElse(k, k)))
        memoizeTableIR(child, childDep, memo)
      case TableToTableApply(child, _) => memoizeTableIR(child, child.typ, memo)
      case MatrixToTableApply(child, _) => memoizeMatrixIR(child, child.typ, memo)
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
      case MatrixUnionCols(left, right) =>
        memoizeMatrixIR(left, requestedType, memo)
        memoizeMatrixIR(right,
          requestedType.copy(globalType = TStruct.empty(),
            rvRowType = requestedType.rvRowType.filterSet((requestedType.rowKey :+ MatrixType.entriesIdentifier).toSet)._1),
          memo)
      case MatrixMapEntries(child, newEntries) =>
        val irDep = memoizeAndGetDep(newEntries, requestedType.entryType, child.typ, memo)
        val depMod = requestedType.copy(rvRowType = TStruct(requestedType.rvRowType.required, requestedType.rvRowType.fields.map { f =>
          if (f.name == MatrixType.entriesIdentifier)
            f.name -> f.typ.asInstanceOf[TArray].copy(elementType = irDep.entryType)
          else
            f.name -> f.typ
        }: _*))
        memoizeMatrixIR(child, unify(child.typ, depMod, irDep), memo)
      case MatrixKeyRowsBy(child, keys, isSorted) =>
        memoizeMatrixIR(child, requestedType.copy(
          rvRowType = unify(child.typ.rvRowType, minimal(child.typ).rvRowType, requestedType.rvRowType),
          rowKey = child.typ.rowKey), memo)
      case MatrixMapRows(child, newRow) =>
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
      case MatrixMapGlobals(child, newGlobals) =>
        val irDep = memoizeAndGetDep(newGlobals, requestedType.globalType, child.typ, memo)
        memoizeMatrixIR(child, unify(child.typ, requestedType.copy(globalType = irDep.globalType), irDep), memo)
      case MatrixRead(_, _, _, _) =>
      case MatrixLiteral(value) =>
      case MatrixChooseCols(child, oldIndices) =>
        memoizeMatrixIR(child, unify(child.typ, requestedType), memo)
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
      case MatrixAggregateRowsByKey(child, entryExpr, rowExpr) =>
        val irDepEntry = memoizeAndGetDep(entryExpr, requestedType.entryType, child.typ, memo)
        val irDepRow = memoizeAndGetDep(rowExpr, requestedType.rowValueStruct, child.typ, memo)
        val childDep = unify(child.typ,
          irDepEntry,
          irDepRow,
          minimal(child.typ).copy(globalType = requestedType.globalType, colType = requestedType.colType)
        )
        memoizeMatrixIR(child, childDep, memo)
      case MatrixAggregateColsByKey(child, entryExpr, colExpr) =>
        val irDepEntry = memoizeAndGetDep(entryExpr, requestedType.entryType, child.typ, memo)
        val irDepCol = memoizeAndGetDep(colExpr, requestedType.colValueStruct, child.typ, memo)
        val rvRowDep = TStruct(
          child.typ.rvRowType.required, child.typ.rvRowType.fields.flatMap { f =>
            if (f.name == MatrixType.entriesIdentifier)
              Some(f.name -> irDepEntry.entryArrayType)
            else {
              val requestedFieldDep = requestedType.rvRowType.fieldOption(f.name).map(_.typ)
              val irFieldDep = irDepEntry.rvRowType.fieldOption(f.name).map(_.typ)
              if (requestedFieldDep.isEmpty && irFieldDep.isEmpty)
                None
              else
                Some(f.name -> unifySeq(f.typ, (requestedFieldDep.iterator ++ irFieldDep.iterator).toFastSeq))
            }
          }: _*
        )
        val childDep = child.typ.copy(
          globalType = unify(child.typ.globalType, irDepEntry.globalType, irDepCol.globalType, requestedType.globalType),
          rvRowType = rvRowDep,
          colType = unify(child.typ.colType, irDepEntry.colType, irDepCol.colType)
        )
        memoizeMatrixIR(child, childDep, memo)
      case TableToMatrixTable(child, rowKey, colKey, rowFields, colFields, nPartitions) =>
        val dependencyMap = (requestedType.rowType.fields.map(f => f.name -> f.typ) ++
          requestedType.colType.fields.map(f => f.name -> f.typ) ++
          requestedType.entryType.fields.map(f => f.name -> f.typ)).toMap
        val rowDep = TStruct(child.typ.rowType.fields.flatMap { f =>
         dependencyMap.get(f.name).map(t => f.name -> t)
        }: _*)
        val requestedChildType = child.typ.copy(
          rowType = unify(child.typ.rowType, rowDep),
          globalType = requestedType.globalType)
        memoizeTableIR(child, requestedChildType, memo)
      case MatrixAnnotateRowsTable(child, table, root) =>
        val fieldDep = requestedType.rvRowType.fieldOption(root).map(_.typ.asInstanceOf[TStruct])
        fieldDep match {
          case Some(struct) =>
            val tableDep = table.typ.copy(rowType = unify(
              table.typ.rowType,
              FastIndexedSeq[TStruct](table.typ.rowType.filterSet(table.typ.key.toSet, true)._1) ++
                FastIndexedSeq(struct): _*),
              globalType = minimal(table.typ.globalType))
            memoizeTableIR(table, tableDep, memo)
            val matDep = requestedType.copy(rvRowType =
              unify(child.typ.rvRowType,
                minimal(child.typ).rvRowType,
                requestedType.rvRowType.filterSet(Set(root), include = false)._1))
            memoizeMatrixIR(child, matDep, memo)
          case None =>
            // don't depend on key IR dependencies if we are going to elide the node anyway
            memoizeMatrixIR(child, requestedType, memo)
        }
      case MatrixAnnotateColsTable(child, table, uid) =>
        val fieldDep = requestedType.colType.fieldOption(uid).map(_.typ.asInstanceOf[TStruct])
        fieldDep match {
          case Some(struct) =>
            val tableDep = table.typ.copy(rowType = unify(
              table.typ.rowType,
              FastIndexedSeq[TStruct](table.typ.rowType.filterSet(table.typ.key.toSet, true)._1) ++
                FastIndexedSeq(struct): _*))
            memoizeTableIR(table,tableDep, memo)
            val matDep = unify(
              child.typ,
              requestedType.copy(colType =
                unify(child.typ.colType,
                  minimal(child.typ).colType,
                  requestedType.colType.filterSet(Set(uid), include = false)._1)))
            memoizeMatrixIR(child, matDep, memo)
          case None =>
            // don't depend on key IR dependencies if we are going to elide the node anyway
            memoizeMatrixIR(child, requestedType, memo)
        }
      case MatrixExplodeRows(child, path) =>
        def getExplodedField(typ: MatrixType): Type = typ.rowType.queryTyped(path.toList)._1
        val preExplosionFieldType = getExplodedField(child.typ)
        val prunedPreExlosionFieldType = try {
          val t = getExplodedField(requestedType)
          preExplosionFieldType match {
            case ta: TArray => ta.copy(elementType = t)
            case ts: TSet => ts.copy(elementType = t)
          }
        } catch {
          case e: AnnotationPathException => minimal(preExplosionFieldType)
        }
        val dep = requestedType.copy(rvRowType = unify(child.typ.rvRowType,
          requestedType.rvRowType.insert(prunedPreExlosionFieldType, path.toList)._1.asInstanceOf[TStruct]))
        memoizeMatrixIR(child, dep, memo)
      case MatrixExplodeCols(child, path) =>
        def getExplodedField(typ: MatrixType): Type = typ.colType.queryTyped(path.toList)._1
        val preExplosionFieldType = getExplodedField(child.typ)
        val prunedPreExplosionFieldType = try {
          val t = getExplodedField(requestedType)
          preExplosionFieldType  match {
            case ta: TArray => ta.copy(elementType = t)
            case ts: TSet => ts.copy(elementType = t)
          }
        } catch {
          case e: AnnotationPathException => minimal(preExplosionFieldType)
        }
        val dep = requestedType.copy(colType = unify(child.typ.colType,
          requestedType.colType.insert(prunedPreExplosionFieldType, path.toList)._1.asInstanceOf[TStruct]))
        memoizeMatrixIR(child, dep, memo)
      case MatrixRepartition(child, _, _) =>
        memoizeMatrixIR(child, requestedType, memo)
      case MatrixUnionRows(children) =>
        children.foreach(memoizeMatrixIR(_, requestedType, memo))
      case MatrixDistinctByRow(child) =>
        memoizeMatrixIR(child, requestedType, memo)
      case CastTableToMatrix(child, entriesFieldName, colsFieldName, _) =>
        val m = Map(MatrixType.entriesIdentifier -> entriesFieldName)
        val childDep = child.typ.copy(
          globalType = unify(child.typ.globalType, requestedType.globalType, TStruct((colsFieldName, TArray(requestedType.colType)))),
          rowType = requestedType.rvRowType.rename(m)
        )
        memoizeTableIR(child, childDep, memo)
      case MatrixToMatrixApply(child, _) => memoizeMatrixIR(child, child.typ, memo)
      case MatrixRename(child, globalMap, colMap, rowMap, entryMap) =>
        val globalMapRev = globalMap.map { case (k, v) => (v, k) }
        val colMapRev = colMap.map { case (k, v) => (v, k) }
        val rowMapRev = rowMap.map { case (k, v) => (v, k) }
        val entryMapRev = entryMap.map { case (k, v) => (v, k) }
        val childDep = MatrixType.fromParts(
          globalType = requestedType.globalType.rename(globalMapRev),
          colType = requestedType.colType.rename(colMapRev),
          colKey = requestedType.colKey.map(k => colMapRev.getOrElse(k, k)),
          rowType = requestedType.rowType.rename(rowMapRev),
          rowKey = requestedType.rowKey.map(k => rowMapRev.getOrElse(k, k)),
          entryType = requestedType.entryType.rename(entryMapRev))
        memoizeMatrixIR(child, childDep, memo)
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
      case ArrayScan(a, zero, accumName, valueName, body) =>
        val aType = a.typ.asInstanceOf[TArray]
        val zeroEnv = memoizeValueIR(zero, zero.typ, memo)
        val bodyEnv = memoizeValueIR(body, body.typ, memo)
        val valueType = bodyEnv.lookupOption(valueName).map(_._2).getOrElse(minimal(-aType.elementType))
        unifyEnvs(
          zeroEnv,
          bodyEnv.delete(accumName).delete(valueName),
          memoizeValueIR(a, aType.copy(elementType = valueType), memo)
        )
      case ArrayLeftJoinDistinct(left, right, l, r, compare, join) =>
        val lType = left.typ.asInstanceOf[TArray]
        val rType = right.typ.asInstanceOf[TArray]

        val compEnv = memoizeValueIR(compare, compare.typ, memo)
        val joinEnv = memoizeValueIR(join, requestedType.asInstanceOf[TArray].elementType, memo)

        val lRequested = unify(lType.elementType,
          compEnv.lookupOption(l).map(_._2).getOrElse(minimal(-lType.elementType)),
          joinEnv.lookupOption(l).map(_._2).getOrElse(minimal(-lType.elementType)))
        val rRequested = unify(rType.elementType,
          compEnv.lookupOption(r).map(_._2).getOrElse(minimal(-rType.elementType)),
          joinEnv.lookupOption(r).map(_._2).getOrElse(minimal(-rType.elementType)))

        unifyEnvs(
          compEnv.delete(l).delete(r),
          joinEnv.delete(l).delete(r),
          memoizeValueIR(left, lType.copy(elementType = lRequested), memo),
          memoizeValueIR(right, rType.copy(elementType = rRequested), memo))

      case ArrayFor(a, valueName, body) =>
        assert(requestedType == TVoid)
        val aType = a.typ.asInstanceOf[TArray]
        val bodyEnv = memoizeValueIR(body, body.typ, memo)
        val valueType = bodyEnv.lookupOption(valueName).map(_._2).getOrElse(minimal(-aType.elementType))
        unifyEnvs(
          bodyEnv.delete(valueName),
          memoizeValueIR(a, aType.copy(elementType = valueType), memo)
        )
      case AggExplode(a, name, body) =>
        val aType = a.typ.asInstanceOf[TArray]
        val bodyDep = memoizeValueIR(body,
          requestedType,
          memo)
        val valueType = bodyDep.lookupOption(name).map(_._2).getOrElse(minimal(-aType.elementType))
        unifyEnvs(
          bodyDep.delete(name),
          memoizeValueIR(a, aType.copy(elementType = valueType), memo)
        )
      case MakeStruct(fields) =>
        val sType = requestedType.asInstanceOf[TStruct]
        unifyEnvsSeq(fields.flatMap { case (fname, fir) =>
          // ignore unreachable fields, these are eliminated on the upwards pass
          sType.fieldOption(fname).map(f => memoizeValueIR(fir, f.typ, memo))
        })
      case InsertFields(old, fields, _) =>
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
      case TableGetGlobals(child) =>
        memoizeTableIR(child, minimal(child.typ).copy(globalType = requestedType.asInstanceOf[TStruct]), memo)
        Env.empty[(Type, Type)]
      case TableCollect(child) =>
        val rStruct = requestedType.asInstanceOf[TStruct]
        val minimalChild = minimal(child.typ)
        memoizeTableIR(child, TableType(
          unify(child.typ.rowType,
            minimalChild.rowType,
            rStruct.fieldOption("rows").map(_.typ.asInstanceOf[TArray].elementType.asInstanceOf[TStruct]).getOrElse(TStruct())),
          minimalChild.key,
          rStruct.fieldOption("global").map(_.typ.asInstanceOf[TStruct]).getOrElse(TStruct())),
          memo)
        Env.empty[(Type, Type)]
      case TableToValueApply(child, _) =>
        memoizeTableIR(child, child.typ, memo)
        Env.empty[(Type, Type)]
      case MatrixToValueApply(child, __) => memoizeMatrixIR(child, child.typ, memo)
        Env.empty[(Type, Type)]
      case TableAggregate(child, query) =>
        val queryDep = memoizeAndGetDep(query, query.typ, child.typ, memo)
        memoizeTableIR(child, queryDep, memo)
        Env.empty[(Type, Type)]
      case MatrixAggregate(child, query) =>
        val queryDep = memoizeAndGetDep(query, query.typ, child.typ, memo)
        memoizeMatrixIR(child, queryDep, memo)
        Env.empty[(Type, Type)]
      case ArrayAgg(a, name, query) =>
        val elemType = a.typ.asInstanceOf[TArray].elementType
        val queryEnv = memoizeValueIR(query, requestedType, memo)
        val requestedElemType = queryEnv.lookupOption(name).map(_._2).getOrElse(minimal(elemType))
        val aEnv = memoizeValueIR(a, TArray(requestedElemType), memo)
        val env = unifyEnvs(queryEnv.delete(name), aEnv)
        env
      case _: IR =>
        val envs = ir.children.flatMap {
          case mir: MatrixIR =>
            memoizeMatrixIR(mir, mir.typ, memo)
            None
          case tir: TableIR =>
            memoizeTableIR(tir, tir.typ, memo)
            None
          case bmir: BlockMatrixIR => //NOTE Currently no BlockMatrixIRs would have dead fields
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
      case TableParallelize(rowsAndGlobal, nPartitions) =>
        TableParallelize(
          upcast(rebuild(rowsAndGlobal, Env.empty[Type], memo),
            memo.lookup(rowsAndGlobal).asInstanceOf[TStruct]),
          nPartitions)
      case TableRead(_, dropRows, tr) => TableRead(dep, dropRows, tr)
      case TableFilter(child, pred) =>
        val child2 = rebuild(child, memo)
        val pred2 = rebuild(pred, child2.typ, memo)
        TableFilter(child2, pred2)
      case TableMapRows(child, newRow) =>
        val child2 = rebuild(child, memo)
        val newRow2 = rebuild(newRow, child2.typ, memo)
        TableMapRows(child2, newRow2)
      case TableMapGlobals(child, newGlobals) =>
        val child2 = rebuild(child, memo)
        TableMapGlobals(child2, rebuild(newGlobals, child2.typ, memo))
      case TableKeyBy(child, keys, isSorted) =>
        var child2 = rebuild(child, memo)
        // fully upcast before shuffle
        if (!isSorted && keys.nonEmpty)
          child2 = upcastTable(child2, memo.lookup(child).asInstanceOf[TableType])
        TableKeyBy(child2, keys, isSorted)
      case TableOrderBy(child, sortFields) =>
        // fully upcast before shuffle
        val child2 = upcastTable(rebuild(child, memo), memo.lookup(child).asInstanceOf[TableType])
        TableOrderBy(child2, sortFields)
      case TableLeftJoinRightDistinct(left, right, root) =>
        if (dep.rowType.hasField(root))
          TableLeftJoinRightDistinct(rebuild(left, memo), rebuild(right, memo), root)
        else
          rebuild(left, memo)
      case TableIntervalJoin(left, right, root) =>
        if (dep.rowType.hasField(root))
          TableIntervalJoin(rebuild(left, memo), rebuild(right, memo), root)
        else
          rebuild(left, memo)
      case TableMultiWayZipJoin(children, fieldName, globalName) =>
        val rebuilt = children.map { c => rebuild(c, memo) }
        val upcasted = rebuilt.map { t => upcastTable(t, memo.lookup(children(0)).asInstanceOf[TableType]) }
        TableMultiWayZipJoin(upcasted, fieldName, globalName)
      case TableAggregateByKey(child, expr) =>
        val child2 = rebuild(child, memo)
        TableAggregateByKey(child2, rebuild(expr, child2.typ, memo))
      case TableKeyByAndAggregate(child, expr, newKey, nPartitions, bufferSize) =>
        val child2 = rebuild(child, memo)
        val expr2 = rebuild(expr, child2.typ, memo)
        val newKey2 = rebuild(newKey, child2.typ, memo)
        TableKeyByAndAggregate(child2, expr2, newKey2, nPartitions, bufferSize)
      case TableRename(child, rowMap, globalMap) =>
        val child2 = rebuild(child, memo)
        TableRename(
          child2,
          rowMap.filterKeys(child2.typ.rowType.hasField),
          globalMap.filterKeys(child2.typ.globalType.hasField))
      case TableUnion(children) =>
        val requestedType = memo.lookup(tir).asInstanceOf[TableType]
        val rebuilt = children.map { c =>
          upcastTable(rebuild(c, memo), requestedType, upcastGlobals = false)
        }
        TableUnion(rebuilt)
      case _ => tir.copy(tir.children.map {
        // IR should be a match error - all nodes with child value IRs should have a rule
        case childT: TableIR => rebuild(childT, memo)
        case childM: MatrixIR => rebuild(childM, memo)
      })
    }
  }

  def rebuild(mir: MatrixIR, memo: Memo[BaseType]): MatrixIR = {
    val requestedType = memo.lookup(mir).asInstanceOf[MatrixType]
    mir match {
      case x@MatrixRead(_, dropCols, dropRows, reader) =>
        MatrixRead(reader.requestType(requestedType), dropCols, dropRows, reader)
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
      case MatrixMapRows(child, newRow) =>
        var child2 = rebuild(child, memo)
        MatrixMapRows(child2, rebuild(newRow, child2.typ, memo))
      case MatrixMapCols(child, newCol, newKey) =>
        // FIXME account for key
        val child2 = rebuild(child, memo)
        MatrixMapCols(child2, rebuild(newCol, child2.typ, memo), newKey)
      case MatrixMapGlobals(child, newGlobals) =>
        val child2 = rebuild(child, memo)
        MatrixMapGlobals(child2, rebuild(newGlobals, child2.typ, memo))
      case MatrixAggregateRowsByKey(child, entryExpr, rowExpr) =>
        val child2 = rebuild(child, memo)
        MatrixAggregateRowsByKey(child2, rebuild(entryExpr, child2.typ, memo), rebuild(rowExpr, child2.typ, memo))
      case MatrixAggregateColsByKey(child, entryExpr, colExpr) =>
        val child2 = rebuild(child, memo)
        MatrixAggregateColsByKey(child2, rebuild(entryExpr, child2.typ, memo), rebuild(colExpr, child2.typ, memo))
      case TableToMatrixTable(child, rowKey, colKey, rowFields, colFields, nPartitions) =>
        val child2 = rebuild(child, memo)
        val childFieldSet = child2.typ.rowType.fieldNames.toSet
        TableToMatrixTable(
          child2,
          rowKey,
          colKey,
          rowFields.filter(childFieldSet.contains),
          colFields.filter(childFieldSet.contains),
          nPartitions)
      case MatrixUnionRows(children) =>
        val requestedType = memo.lookup(mir).asInstanceOf[MatrixType]
        MatrixUnionRows(children.map { child =>
          upcast(rebuild(child, memo), requestedType,
            upcastCols = false,
            upcastGlobals = false)
        } )
      case MatrixAnnotateRowsTable(child, table, root) =>
        // if the field is not used, this node can be elided entirely
        if (!requestedType.rvRowType.hasField(root))
          rebuild(child, memo)
        else {
          val child2 = rebuild(child, memo)
          val table2 = rebuild(table, memo)
          MatrixAnnotateRowsTable(child2, table2, root)
        }
      case MatrixAnnotateColsTable(child, table, uid) =>
        // if the field is not used, this node can be elided entirely
        if (!requestedType.colType.hasField(uid))
          rebuild(child, memo)
        else {
          val child2 = rebuild(child, memo)
          val table2 = rebuild(table, memo)
          MatrixAnnotateColsTable(child2, table2, uid)
        }
      case MatrixRename(child, globalMap, colMap, rowMap, entryMap) =>
        val child2 = rebuild(child, memo)
        MatrixRename(
          child2,
          globalMap.filterKeys(child2.typ.globalType.hasField),
          colMap.filterKeys(child2.typ.colType.hasField),
          rowMap.filterKeys(child2.typ.rowType.hasField),
          entryMap.filterKeys(child2.typ.entryType.hasField))
      case _ => mir.copy(mir.children.map {
        // IR should be a match error - all nodes with child value IRs should have a rule
        case childT: TableIR => rebuild(childT, memo)
        case childM: MatrixIR => rebuild(childM, memo)
      })

    }
  }

  def rebuild(ir: IR, in: BaseType, memo: Memo[BaseType]): IR = {
    rebuild(ir, relationalTypeToEnv(in), memo)
  }

  def rebuild(ir: IR, in: Env[Type], memo: Memo[BaseType]): IR = {
    val requestedType = memo.lookup(ir).asInstanceOf[Type]
    ir match {
      case NA(typ) => NA(requestedType)
      case If(cond, cnsq, alt) =>
        val cond2 = rebuild(cond, in, memo)
        val cnsq2 = rebuild(cnsq, in, memo)
        val alt2 = rebuild(alt, in, memo)
        If.unify(cond2, cnsq2, alt2, unifyType = Some(requestedType))
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
      case ArrayScan(a, zero, accumName, valueName, body) =>
        val a2 = rebuild(a, in, memo)
        val z2 = rebuild(zero, in, memo)
        ArrayScan(
          a2,
          z2,
          accumName,
          valueName,
          rebuild(body, in.bind(accumName -> z2.typ, valueName -> -a2.typ.asInstanceOf[TArray].elementType), memo)
        )
      case ArrayLeftJoinDistinct(left, right, l, r, compare, join) =>
        val left2 = rebuild(left, in, memo)
        val right2 = rebuild(right, in, memo)

        val ltyp = left2.typ.asInstanceOf[TArray]
        val rtyp = right2.typ.asInstanceOf[TArray]
        ArrayLeftJoinDistinct(
          left2, right2, l, r,
          rebuild(compare, in.bind(l -> -ltyp.elementType, r -> -rtyp.elementType), memo),
          rebuild(join, in.bind(l -> -ltyp.elementType, r -> -rtyp.elementType), memo))

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
      case InsertFields(old, fields, fieldOrder) =>
        val depStruct = requestedType.asInstanceOf[TStruct]
        val depFields = depStruct.fieldNames.toSet
        val rebuiltChild = rebuild(old, in, memo)
        val preservedChildFields = rebuiltChild.typ.asInstanceOf[TStruct].fieldNames.toSet
        InsertFields(rebuiltChild,
          fields.flatMap { case (f, fir) =>
            if (depFields.contains(f))
              Some(f -> rebuild(fir, in, memo))
            else {
              log.info(s"Prune: InsertFields: eliminating field '$f'")
              None
            }
          }, fieldOrder.map(fds => fds.filter(f => depFields.contains(f) || preservedChildFields.contains(f))))
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
      case MatrixAggregate(child, query) =>
        val child2 = rebuild(child, memo)
        val query2 = rebuild(query, child2.typ, memo)
        MatrixAggregate(child2, query2)
      case TableCollect(child) =>
        val rStruct = requestedType.asInstanceOf[TStruct]
        if (!rStruct.hasField("rows"))
          if (rStruct.hasField("global"))
            MakeStruct(FastSeq("global" -> TableGetGlobals(rebuild(child, memo))))
          else
            MakeStruct(FastSeq())
        else
          TableCollect(rebuild(child, memo))
      case ArrayAgg(a, name, query) =>
        val a2 = rebuild(a, in, memo)
        ArrayAgg(
          a2,
          name,
          rebuild(query, in.bind(name, a2.typ.asInstanceOf[TArray].elementType), memo)
        )
      case _ =>
        ir.copy(ir.children.map {
          case valueIR: IR => rebuild(valueIR, in, memo)
          case mir: MatrixIR => rebuild(mir, memo)
          case tir: TableIR => rebuild(tir, memo)
          case bmir: BlockMatrixIR => bmir //NOTE Currently no BlockMatrixIRs would have dead fields
        })
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
          val ref = Ref(uid, -ta.elementType)
          ArrayMap(ir, uid, upcast(ref, ra.elementType))
        case tt: TTuple =>
          val rt = rType.asInstanceOf[TTuple]
          val uid = genUID()
          val ref = Ref(uid, ir.typ)
          val mt = MakeTuple(rt.types.zipWithIndex.map { case (typ, i) =>
              upcast(GetTupleElement(ref, i), typ)
          })
          Let(uid, ir, mt)
        case td: TDict =>
          val rd = rType.asInstanceOf[TDict]
          ToDict(upcast(ToArray(ir), TArray(rd.elementType)))
        case ts: TSet =>
          val rs = rType.asInstanceOf[TSet]
          ToSet(upcast(ToArray(ir), TSet(rs.elementType)))
        case t => ir
      }
    }
  }

  def upcast(ir: MatrixIR, rType: MatrixType,
    upcastRows: Boolean = true,
    upcastCols: Boolean = true,
    upcastGlobals: Boolean = true,
    upcastEntries: Boolean = true): MatrixIR = {

    if (ir.typ == rType || !(upcastRows || upcastCols || upcastGlobals || upcastEntries))
      ir
    else {
      var mt = ir
      if (upcastEntries && mt.typ.entryType != rType.entryType)
        mt = MatrixMapEntries(mt, upcast(Ref("g", mt.typ.entryType), rType.entryType))

      if (upcastRows && mt.typ.rowType != rType.rowType)
        mt = MatrixMapRows(mt, upcast(Ref("va", mt.typ.rvRowType), rType.rvRowType))

      if (upcastCols && mt.typ.colType != rType.colType)
        mt = MatrixMapCols(mt, upcast(Ref("sa", mt.typ.colType), rType.colType), None)

      if (upcastGlobals && mt.typ.globalType != rType.globalType)
        mt = MatrixMapGlobals(mt, upcast(Ref("global", ir.typ.globalType), rType.globalType))

      mt
    }
  }

  def upcastTable(
    ir: TableIR,
    rType: TableType,
    upcastRow: Boolean = true,
    upcastGlobals: Boolean = true
  ): TableIR = {
    if (ir.typ == rType)
      ir
    else {
      var table = ir
      if (upcastRow && ir.typ.rowType != rType.rowType) {
        table = TableMapRows(table, upcast(Ref("row", table.typ.rowType), rType.rowType))
      }
      if (upcastGlobals && ir.typ.globalType != rType.globalType) {
        table = TableMapGlobals(table,
          upcast(Ref("global", table.typ.globalType), rType.globalType))
      }
      table
    }
  }
}

