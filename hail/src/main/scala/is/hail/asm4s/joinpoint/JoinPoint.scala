package is.hail.asm4s.joinpoint

// uninhabitable dummy type, indicating some sort of control flow rather than returning a value;
// used as the return type of JoinPoints
case class Ctrl(n: Nothing)
