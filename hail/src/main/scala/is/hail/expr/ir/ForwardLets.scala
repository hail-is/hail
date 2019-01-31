package is.hail.expr.ir

case class RefCounter(var maxNestingLevel: Int = 0, var nRef: Int = 0) {
  def register(level: Int) {
    maxNestingLevel = math.max(level, maxNestingLevel)
    nRef += 1
  }
}

object ForwardLets {
  def analyzeRefsByNestingLevel(ir: BaseIR, binding: String, nestingLevel: Int, counter: RefCounter): Unit = {
    ir match {
      case Ref(`binding`, _) =>
        counter.register(nestingLevel)
      case ArrayMap(a, name, body) =>
        analyzeRefsByNestingLevel(a, binding, nestingLevel, counter)
        if (name != binding)
          analyzeRefsByNestingLevel(body, binding, nestingLevel + 1, counter)
      case ArrayFilter(a, name, body) =>
        analyzeRefsByNestingLevel(a, binding, nestingLevel, counter)
        if (name != binding)
          analyzeRefsByNestingLevel(body, binding, nestingLevel + 1, counter)
      case ArrayFlatMap(a, name, body) =>
        analyzeRefsByNestingLevel(a, binding, nestingLevel, counter)
        if (name != binding)
          analyzeRefsByNestingLevel(body, binding, nestingLevel + 1, counter)
      case ArrayFor(a, name, body) =>
        analyzeRefsByNestingLevel(a, binding, nestingLevel, counter)
        if (name != binding)
          analyzeRefsByNestingLevel(body, binding, nestingLevel + 1, counter)
      case ArrayAgg(a, name, query) =>
        analyzeRefsByNestingLevel(a, binding, nestingLevel, counter)
        if (name != binding)
          analyzeRefsByNestingLevel(query, binding, nestingLevel + 1, counter)
      case ArrayFold(a, zero, accumName, valueName, body) =>
        analyzeRefsByNestingLevel(a, binding, nestingLevel, counter)
        analyzeRefsByNestingLevel(zero, binding, nestingLevel, counter)
        if (accumName != binding && valueName != binding)
          analyzeRefsByNestingLevel(body, binding, nestingLevel + 1, counter)
      case ArrayScan(a, zero, accumName, valueName, body) =>
        analyzeRefsByNestingLevel(a, binding, nestingLevel, counter)
        analyzeRefsByNestingLevel(zero, binding, nestingLevel, counter)
        if (accumName != binding && valueName != binding)
          analyzeRefsByNestingLevel(body, binding, nestingLevel + 1, counter)
      case AggExplode(array, name, aggBody) =>
        analyzeRefsByNestingLevel(array, binding, nestingLevel, counter)
        if (name != binding)
          analyzeRefsByNestingLevel(aggBody, binding, nestingLevel + 1, counter)
      case _: AggFilter | _: AggGroupBy | _: ApplyAggOp | _: ArrayAgg =>
        ir.children.foreach(analyzeRefsByNestingLevel(_, binding, nestingLevel + 1, counter))
      case _ =>
        ir.children.foreach(analyzeRefsByNestingLevel(_, binding, nestingLevel, counter))
    }
  }

  def apply(ir: BaseIR): BaseIR = {

    RewriteBottomUp(ir, {
      case let@Let(binding, value, letBody) =>
        value match {
          case _: Ref =>
            Some(Subst(letBody, Env(binding -> value)))
          case _ =>
            val counter = RefCounter()
            analyzeRefsByNestingLevel(letBody, binding, 0, counter)

            if (counter.nRef == 0)
              Some(letBody)
            else if (counter.nRef == 1 && counter.maxNestingLevel == 0)
              Some(Subst(letBody, Env(binding -> value)))
            else
              None
        }
      case _ => None
    })
  }
}
