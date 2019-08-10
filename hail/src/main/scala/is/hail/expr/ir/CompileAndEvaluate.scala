package is.hail.expr.ir

import is.hail.annotations.{Region, RegionValueBuilder, SafeRow}
import is.hail.expr.types.physical.{PBaseStruct, PType}
import is.hail.expr.types.virtual.{TStruct, TTuple, TVoid, Type}
import is.hail.utils.{ExecutionTimer, FastIndexedSeq, FastSeq, Timings}
import org.apache.spark.sql.Row

object CompileAndEvaluate {
  def apply[T](ctx: ExecuteContext,
    ir0: IR,
    env: Env[(Any, Type)] = Env(),
    args: IndexedSeq[(Any, Type)] = FastIndexedSeq(),
    optimize: Boolean = true
  ): (T, Timings) = {
    val evalContext = "CompileAndEvaluate"
    val timer = new ExecutionTimer(evalContext)
    var ir = ir0

    def optimizeIR(canGenerateLiterals: Boolean, context: String) {
      ir = timer.time(Optimize(ir, noisy = true, canGenerateLiterals, Some(s"$evalContext: $context")), context)
      TypeCheck(ir, BindingEnv(env.mapValues(_._2)))
    }

    if (optimize) optimizeIR(true, "first pass")
    ir = LowerMatrixIR(ir)
    if (optimize) optimizeIR(true, "after Matrix lowering")
    ir = EvaluateRelationalLets(ir).asInstanceOf[IR]
    ir = LiftNonCompilable(ir).asInstanceOf[IR]

    // void is not really supported by IR utilities
    if (ir.typ == TVoid) {
      val res = timer.time(Interpret[T](ctx, ir, env, args, None, optimize = false), "interpret")
      return (res, timer.timings)
    }

    val (evalIR, ncValue, ncType, ncVar) = timer.time(InterpretNonCompilable(ctx, ir), "interpret non-compilable")
    ir = evalIR

    val argsInVar = genUID()
    val argsInType = TTuple(args.map(_._2): _*)
    val argsInValue = Row.fromSeq(args.map(_._1))

    // don't do extra work
    val rewriteArgsIn: IR => IR = if (args.isEmpty) identity[IR] else {
      def rewriteArgsIn(x: IR): IR = {
        x match {
          case In(i, t) =>
            GetTupleElement(Ref(argsInVar, argsInType), i)
          case _ =>
            MapIR(rewriteArgsIn)(x)
        }
      }
      rewriteArgsIn
    }

    val (envVar, envType, envValue, rewriteEnv): (String, TStruct, Any, IR => IR) = {
      env.m.toArray match {
        // common case; don't do extra work
        case Array((envVar, (envValue, envType: TStruct))) => (envVar, envType, envValue, identity[IR])
        case eArray =>
          val envVar = genUID()
          val envType = TStruct(eArray.map { case (name, (_, t)) => name -> t}: _*)
          val envValue = Row.fromSeq(eArray.map(_._2._1))
          (envVar,
            envType,
            envValue,
            Subst(_, BindingEnv(Env[IR](envType.fieldNames.map(s => s -> GetField(Ref(envVar, envType), s)): _*))))
      }
    }

    ir = rewriteArgsIn(ir)
    ir = rewriteEnv(ir)

    val ncPType = PType.canonical(ncType)
    val argsInPType = PType.canonical(argsInType)
    val envPType = PType.canonical(envType)

    val (resultPType, f) = timer.time(Compile[Long, Long, Long, Long](
      ncVar, ncPType,
      argsInVar, argsInPType,
      envVar, envPType,
      MakeTuple.ordered(FastSeq(ir))), "compile")

    val value = timer.time(
      Region.scoped { region =>
        val rvb = new RegionValueBuilder(region)
        rvb.start(ncPType)
        rvb.addAnnotation(ncType, ncValue)
        val ncOffset = rvb.end()

        rvb.start(argsInPType)
        rvb.addAnnotation(argsInType, argsInValue)
        val argsInOffset = rvb.end()

        rvb.start(envPType)
        rvb.addAnnotation(envType, envValue)
        val envOffset = rvb.end()

        val resultOff = f(0, region)(region,
          ncOffset, ncValue == null,
          argsInOffset, argsInValue == null,
          envOffset, envValue == null)
        SafeRow(resultPType.asInstanceOf[PBaseStruct], region, resultOff).getAs[T](0)
      },
      "runtime")

    (value, timer.timings)
  }
}
