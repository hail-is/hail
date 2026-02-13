package is.hail.expr.ir

import is.hail.backend.ExecuteContext
import is.hail.expr.ir.NormalizeNames.needsRenaming
import is.hail.expr.ir.defs._
import is.hail.types.virtual.Type
import is.hail.utils.StackSafe._

import scala.annotation.tailrec

object NormalizeNames {
  def apply[T <: BaseIR](allowFreeVariables: Boolean = false)(ctx: ExecuteContext, ir: T): T =
    ctx.time {
      val freeVariables: Set[Name] = ir match {
        case ir: IR =>
          if (allowFreeVariables) {
            val env = FreeVariables(ir, true, true)
            env.eval.m.keySet union
              env.agg.map(_.m.keySet).getOrElse(Set.empty) union
              env.scan.map(_.m.keySet).getOrElse(Set.empty) union
              env.relational.m.keySet
          } else {
            Set.empty
          }
        case _ =>
          Set.empty
      }
      val env = BindingEnv.empty[Name].createAgg.createScan
      val noSharing = ir.noSharing(ctx)
      val normalize = new NormalizeNames(freeVariables)
      val res = normalize.normalizeIR(noSharing, env).run().asInstanceOf[T]
      uidCounter = math.max(uidCounter, normalize.count.toLong)
      res
    }

  protected def needsRenaming(ir: BaseIR): Boolean = ir match {
    case _: RelationalLetMatrixTable | _: TableGen | _: TableMapPartitions | _: RelationalLetTable =>
      true
    case _: MatrixIR | _: TableIR =>
      false
    case _: TableAggregate | _: MatrixAggregate =>
      false
    case _: IR | _: BlockMatrixIR =>
      true
  }
}

class NormalizeNames(freeVariables: Set[Name]) {
  var count: Int = 0

  @tailrec private def gen(): Name = {
    count += 1
    val name = Name(s"__norm_$count")
    if (freeVariables.contains(name)) {
      gen()
    } else {
      name
    }
  }

  private def normalizeIR(ir: BaseIR, env: BindingEnv[Name]): StackFrame[BaseIR] = ir match {
    case Block(bindings, body) =>
      var newEnv = env
      for {
        newBindings <- bindings.mapRecur { case Binding(name, value, scope) =>
          for (newValue <- normalizeIR(value, newEnv.promoteScope(scope))) yield {
            val newName = gen()
            newEnv = newEnv.bindInScope(name, newName, scope)
            Binding(newName, newValue.asInstanceOf[IR], scope)
          }
        }
        newBody <- normalizeIR(body, newEnv)
      } yield Block(newBindings, newBody.asInstanceOf[IR])
    case Ref(name, typ) =>
      val newName = env.eval.lookupOption(name).getOrElse {
        if (!freeVariables.contains(name)) throw new RuntimeException(
          s"found free variable in normalize: $name; ${env.pretty(x => x.str)}"
        )
        else name
      }
      done(Ref(newName, typ))
    case RelationalRef(name, typ) =>
      val newName = env.relational.lookupOption(name).getOrElse(
        if (!freeVariables.contains(name)) throw new RuntimeException(
          s"found free variable in normalize: $name; ${env.pretty(x => x.str)}"
        )
        else name
      )
      done(RelationalRef(newName, typ))
    case Recur(name, args, typ) =>
      val newName = env.eval.lookupOption(name) match {
        case Some(n) => n
        case None =>
          if (!freeVariables.contains(name))
            throw new RuntimeException(s"found free loop variable in normalize: $name")
          else
            name
      }
      Recur(newName, args, typ).mapChildrenStackSafe(normalizeIR(_, env))
    case ir =>
      val bindingsMap = is.hail.collection.compat.mutable.AnyRefMap.empty[Name, Name]
      val updateEnv: (BindingEnv[Name], Bindings[Type]) => BindingEnv[Name] =
        if (needsRenaming(ir)) { (env, bindings) =>
          val bindingsNames = bindings.map((name, _) => bindingsMap.getOrElseUpdate(name, gen()))
          env.extend(bindingsNames)
        } else { (env, bindings) => env.extend(bindings.map((name, _) => name)) }
      ir.mapChildrenWithEnvStackSafe(env, updateEnv)(normalizeIR).map(modifyNames(_, bindingsMap))
  }

  private def modifyNames(ir: BaseIR, map: collection.Map[Name, Name]): BaseIR = {
    def rename(name: Name): Name = map.getOrElse(name, name)

    ir match {
      case TailLoop(name, args, resultType, body) =>
        TailLoop(rename(name), args.map { case (name, ir) => (rename(name), ir) }, resultType, body)
      case x: StreamMap => x.copy(name = rename(x.name))
      case x: StreamZip => x.copy(names = x.names.map(rename))
      case x: StreamZipJoin => x.copy(curKey = rename(x.curKey), curVals = rename(x.curVals))
      case x: StreamZipJoinProducers =>
        x.copy(ctxName = rename(x.ctxName), curKey = rename(x.curKey), curVals = rename(x.curVals))
      case x: StreamLeftIntervalJoin =>
        x.copy(lname = rename(x.lname), rname = rename(x.rname))
      case x: StreamFor => x.copy(valueName = rename(x.valueName))
      case x: StreamFlatMap => x.copy(name = rename(x.name))
      case x: StreamFilter => x.copy(name = rename(x.name))
      case x: StreamTakeWhile => x.copy(elementName = rename(x.elementName))
      case x: StreamDropWhile => x.copy(elementName = rename(x.elementName))
      case x: StreamFold => x.copy(accumName = rename(x.accumName), valueName = rename(x.valueName))
      case x: StreamFold2 => x.copy(
          accum = x.accum.map { case (name, ir) => (rename(name), ir) },
          valueName = rename(x.valueName),
        )
      case x: StreamBufferedAggregate => x.copy(name = rename(x.name))
      case x: RunAggScan => x.copy(name = rename(x.name))
      case x: StreamAgg => x.copy(name = rename(x.name))
      case x: StreamScan => x.copy(accumName = rename(x.accumName), valueName = rename(x.valueName))
      case x: StreamAggScan => x.copy(name = rename(x.name))
      case x: StreamJoinRightDistinct => x.copy(l = rename(x.l), r = rename(x.r))
      case x: ArraySort => x.copy(left = rename(x.left), right = rename(x.right))
      case x: ArrayMaximalIndependentSet =>
        x.copy(tieBreaker = x.tieBreaker.map { case (l, r, f) => (rename(l), rename(r), f) })
      case x: AggArrayPerElement =>
        x.copy(indexName = rename(x.indexName), elementName = rename(x.elementName))
      case x: AggFold =>
        x.copy(accumName = rename(x.accumName), otherAccumName = rename(x.otherAccumName))
      case x: NDArrayMap => x.copy(valueName = rename(x.valueName))
      case x: NDArrayMap2 => x.copy(lName = rename(x.lName), rName = rename(x.rName))
      case x: CollectDistributedArray => x.copy(cname = rename(x.cname), gname = rename(x.gname))
      case x: AggExplode => x.copy(name = rename(x.name))
      case x: RelationalLet => x.copy(name = rename(x.name))
      case x: RelationalLetTable => x.copy(name = rename(x.name))
      case x: RelationalLetMatrixTable => x.copy(name = rename(x.name))
      case x: TableMapPartitions => x.copy(
          globalName = rename(x.globalName),
          partitionStreamName = rename(x.partitionStreamName),
        )
      case x: TableGen => x.copy(cname = rename(x.cname), gname = rename(x.gname))
      case x: BlockMatrixMap => x.copy(eltName = rename(x.eltName))
      case x: BlockMatrixMap2 =>
        x.copy(leftName = rename(x.leftName), rightName = rename(x.rightName))

      case x => x
    }
  }
}
