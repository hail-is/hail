package is.hail

object ExecStrategy extends Enumeration {
  type ExecStrategy = Value
  val Interpret, InterpretUnoptimized, JvmCompile, LoweredJVMCompile, JvmCompileUnoptimized = Value

  val unoptimizedCompileOnly: Set[ExecStrategy] = Set(JvmCompileUnoptimized)
  val compileOnly: Set[ExecStrategy] = Set(JvmCompile, JvmCompileUnoptimized)

  val javaOnly: Set[ExecStrategy] =
    Set(Interpret, InterpretUnoptimized, JvmCompile, JvmCompileUnoptimized)

  val interpretOnly: Set[ExecStrategy] = Set(Interpret, InterpretUnoptimized)

  val nonLowering: Set[ExecStrategy] =
    Set(Interpret, InterpretUnoptimized, JvmCompile, JvmCompileUnoptimized)

  val lowering: Set[ExecStrategy] = Set(LoweredJVMCompile)
  val backendOnly: Set[ExecStrategy] = Set(LoweredJVMCompile)
  val allRelational: Set[ExecStrategy] = interpretOnly.union(lowering)
}
